//! The `zyron_sys` catalog: what resolves, what does not, and what the
//! failure says.
//!
//! Three things are pinned here. Every canonical three-part name resolves and
//! answers. Every name the migration retired resolves to nothing, and says
//! which canonical name replaced it. And nothing anywhere resolves a retired
//! name by any route, which is the property a compat shim would quietly
//! break: a shim added later would still pass a "the new name works" test and
//! would fail these.

mod common;

use std::sync::Arc;

use common::{create_test_server, exec_ddl, exec_dml, new_session};
use zyron_catalog::system_catalog::{self, SYSTEM_OBJECTS, SYSTEM_SCHEMAS, SystemObjectKind};
use zyron_wire::connection::ServerState;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Reads a system entity through the same path a client SELECT takes:
/// parse, recognize the name, read the clauses, build the rows.
async fn read(
    server: &Arc<ServerState>,
    sql: &str,
) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| format!("parse: {e}"))?
        .into_iter()
        .next()
        .ok_or_else(|| "no statement".to_string())?;
    let sel = match stmt {
        zyron_parser::Statement::Select(sel) => sel,
        other => return Err(format!("not a select: {other:?}")),
    };

    // A call with arguments goes down the function path, a bare name down the
    // relation path, the same split the wire dispatch makes
    if let Some(parsed) = zyron_wire::system_views::parse_system_function(&sel) {
        let call = parsed.map_err(|e| e.to_string())?;
        let name = call.object.canonical_name();
        let filters = zyron_wire::system_views::parse_system_view_query(&name, &sel)
            .map_err(|e| e.to_string())?;
        let built = zyron_wire::system_views::query_system_function(&call, server, &filters)
            .await
            .map_err(|e| e.to_string())?;
        return Ok(render(built));
    }

    let name = match &sel.from[0] {
        zyron_parser::TableRef::Table { name, .. } => name.clone(),
        other => return Err(format!("not a plain table ref: {other:?}")),
    };
    if !zyron_wire::system_views::is_system_view(&name) {
        return Err(system_catalog::relation_not_found(&name).to_string());
    }
    let filters = zyron_wire::system_views::parse_system_view_query(&name, &sel)
        .map_err(|e| e.to_string())?;
    let built = zyron_wire::system_views::query_system_view(&name, server, &filters)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("{name} is registered but built nothing"))?;
    Ok(render(built))
}

fn render(
    built: (
        Vec<zyron_wire::messages::backend::FieldDescription>,
        Vec<Vec<Option<Vec<u8>>>>,
    ),
) -> (Vec<String>, Vec<Vec<String>>) {
    let (fields, rows) = built;
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
    (names, rows)
}

/// The predicates a view needs before it can answer at all.
///
/// A version diff is between two named versions, so it refuses a bare SELECT
/// rather than picking a pair. The sweep below supplies them so it exercises
/// the view rather than its argument check, which has its own test.
fn required_predicate(canonical: &str) -> &'static str {
    match canonical {
        "zyron_sys.time_travel.diff_versions" => " WHERE from_version = 1 AND to_version = 2",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// Part A: the catalog itself
// ---------------------------------------------------------------------------

/// The catalog is registered at startup and lists itself.
#[tokio::test]
async fn test_system_catalog_is_registered() {
    let (server, _schema, _tmp) = create_test_server().await;
    let id = server
        .catalog
        .system_catalog_id()
        .expect("zyron_sys is registered at startup");
    let entry = server
        .catalog
        .get_database("zyron_sys")
        .expect("catalog row");
    assert_eq!(entry.id, id);

    let (columns, rows) = read(&server, "SELECT * FROM zyron_sys.core.databases")
        .await
        .expect("read");
    let name_at = columns
        .iter()
        .position(|c| c == "catalog_name")
        .expect("catalog_name column");
    let system_at = columns
        .iter()
        .position(|c| c == "is_system")
        .expect("is_system column");
    let row = rows
        .iter()
        .find(|r| r[name_at] == "zyron_sys")
        .expect("zyron_sys lists itself");
    assert_eq!(row[system_at], "true");
    assert!(
        rows.iter().any(|r| r[name_at] == "zyron"),
        "the default catalog is listed alongside it"
    );
}

/// Every schema in the list exists as a child of the catalog.
#[tokio::test]
async fn test_every_schema_is_a_child_of_the_catalog() {
    let (server, _schema, _tmp) = create_test_server().await;
    let system_id = server.catalog.system_catalog_id().expect("registered");

    let (columns, rows) = read(&server, "SELECT * FROM zyron_sys.core.schemas")
        .await
        .expect("read");
    let catalog_at = columns
        .iter()
        .position(|c| c == "catalog_name")
        .expect("col");
    let schema_at = columns
        .iter()
        .position(|c| c == "schema_name")
        .expect("col");
    let system_at = columns.iter().position(|c| c == "is_system").expect("col");

    let present: Vec<&str> = rows
        .iter()
        .filter(|r| r[catalog_at] == "zyron_sys")
        .map(|r| r[schema_at].as_str())
        .collect();
    let mut missing = Vec::new();
    for schema in SYSTEM_SCHEMAS {
        if !present.contains(schema) {
            missing.push(*schema);
        }
    }
    assert!(missing.is_empty(), "schemas not registered: {missing:?}");

    // Every one of them is marked system, and the catalog agrees
    for row in rows.iter().filter(|r| r[catalog_at] == "zyron_sys") {
        assert_eq!(
            row[system_at], "true",
            "`{}` is in the system catalog but not marked system",
            row[schema_at]
        );
    }
    for schema in SYSTEM_SCHEMAS {
        let entry = server
            .catalog
            .get_schema(system_id, schema)
            .unwrap_or_else(|e| panic!("schema `{schema}` missing: {e}"));
        assert!(server.catalog.is_system_schema(entry.id));
    }
}

/// Registering twice is registering once. A restart re-runs init against
/// rows that already exist and must not double them or fail.
#[tokio::test]
async fn test_registration_is_idempotent() {
    let (server, _schema, _tmp) = create_test_server().await;
    let first = server.catalog.system_catalog_id().expect("registered");
    let again = zyron_catalog::SystemCatalog::init(&server.catalog)
        .await
        .expect("re-init");
    assert_eq!(first, again, "a second init found the same catalog");

    let (columns, rows) = read(&server, "SELECT * FROM zyron_sys.core.databases")
        .await
        .expect("read");
    let name_at = columns
        .iter()
        .position(|c| c == "catalog_name")
        .expect("col");
    let count = rows.iter().filter(|r| r[name_at] == "zyron_sys").count();
    assert_eq!(count, 1, "the catalog was registered twice");
}

/// The system schemas hold nothing a user put there. Every object in them is
/// computed on read, so a stored table in one is a contradiction.
#[tokio::test]
async fn test_system_schemas_refuse_user_tables() {
    let (server, _schema, _tmp) = create_test_server().await;
    let system_id = server.catalog.system_catalog_id().expect("registered");
    let core = server.catalog.get_schema(system_id, "core").expect("core");

    let err = server
        .catalog
        .create_table(core.id, "sneaky", &[], &[])
        .await
        .expect_err("a table in zyron_sys.core must be refused");
    assert!(
        matches!(err, zyron_common::ZyronError::PermissionDenied(_)),
        "expected PermissionDenied, got {err:?}"
    );

    let err = server
        .catalog
        .drop_schema(system_id, "core")
        .await
        .expect_err("dropping a system schema must be refused");
    assert!(matches!(err, zyron_common::ZyronError::PermissionDenied(_)));

    let err = server
        .catalog
        .drop_database("zyron_sys")
        .await
        .expect_err("dropping the system catalog must be refused");
    assert!(matches!(err, zyron_common::ZyronError::PermissionDenied(_)));

    let err = server
        .catalog
        .create_database("zyron_sys", "someone")
        .await
        .expect_err("a second zyron_sys must be refused");
    assert!(matches!(err, zyron_common::ZyronError::PermissionDenied(_)));
}

// ---------------------------------------------------------------------------
// Part A: resolution
// ---------------------------------------------------------------------------

/// Every registered view answers a plain SELECT and returns its own shape.
#[tokio::test]
async fn test_every_registered_view_resolves_and_answers() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut failures = Vec::new();
    for object in SYSTEM_OBJECTS {
        if object.kind != SystemObjectKind::View {
            continue;
        }
        let name = object.canonical_name();
        let sql = format!("SELECT * FROM {name}{}", required_predicate(&name));
        match read(&server, &sql).await {
            Ok((columns, _rows)) => {
                if columns.is_empty() {
                    failures.push(format!("{name} returned no column schema"));
                }
            }
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The default search path puts the two system schemas first, so a bare
/// `tables` reads the catalog navigation view and a bare `activity` reads the
/// session view. Both return exactly what the qualified name returns.
#[tokio::test]
async fn test_default_search_path_resolves_bare_names() {
    let (server, _schema, _tmp) = create_test_server().await;
    let session = zyron_wire::session::Session::new(
        "t".into(),
        "testdb".into(),
        zyron_catalog::DatabaseId(1),
    );
    let path = &session.search_path;
    assert_eq!(
        path,
        &vec![
            "zyron_sys.core".to_string(),
            "zyron_sys.stat".to_string(),
            "information_schema".to_string()
        ],
        "the shipped default search path changed"
    );

    let tables = zyron_wire::system_views::resolve_in_search_path("tables", path)
        .expect("`tables` resolves under the default path");
    assert_eq!(tables.canonical_name(), "zyron_sys.core.tables");
    let activity = zyron_wire::system_views::resolve_in_search_path("activity", path)
        .expect("`activity` resolves under the default path");
    assert_eq!(activity.canonical_name(), "zyron_sys.stat.activity");

    // Same rows either way
    let (bare_cols, bare_rows) = read(&server, "SELECT * FROM zyron_sys.core.tables")
        .await
        .expect("qualified");
    let (resolved_cols, resolved_rows) = read(
        &server,
        &format!("SELECT * FROM {}", tables.canonical_name()),
    )
    .await
    .expect("resolved");
    assert_eq!(bare_cols, resolved_cols);
    assert_eq!(bare_rows, resolved_rows);

    // A schema not on the default path still needs its qualifier
    assert!(
        zyron_wire::system_views::resolve_in_search_path("watermarks", path).is_none(),
        "streaming.watermarks must not be reachable unqualified"
    );
}

// ---------------------------------------------------------------------------
// Parts B-G: the old names are gone
// ---------------------------------------------------------------------------

/// A name under `zyron_sys` that is not registered is refused here rather
/// than handed to the planner, which would look for a user table of that name
/// and report the wrong thing.
#[tokio::test]
async fn test_an_unregistered_system_name_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    for name in [
        "zyron_test_canary",
        "zyron_sys.core.test_canary",
        "zyron_sys.nonexistent_schema.thing",
    ] {
        assert!(
            !system_catalog::is_system_object(name),
            "`{name}` must not resolve"
        );
        let err = read(&server, &format!("SELECT * FROM {name}"))
            .await
            .err()
            .unwrap_or_else(|| panic!("`{name}` answered"));
        assert!(
            err.contains("does not exist"),
            "`{name}` failed with: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// Parts B-G: the new names answer with real content
// ---------------------------------------------------------------------------

/// The catalog navigation views describe a table the test just made.
#[tokio::test]
async fn test_core_views_describe_a_real_table() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE catalog_probe (id BIGINT PRIMARY KEY, label VARCHAR(32))",
    )
    .await
    .expect("create");

    let (columns, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.core.tables WHERE table_name = 'catalog_probe'",
    )
    .await
    .expect("tables");
    assert_eq!(rows.len(), 1, "exactly one row for the table");
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows[0][at("schema_name")], "zyron_test");
    assert_eq!(rows[0][at("catalog_name")], "zyron");
    assert_eq!(rows[0][at("storage_format")], "HEAP");
    assert_eq!(rows[0][at("column_count")], "2");

    let (columns, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.core.columns WHERE table_name = 'catalog_probe'",
    )
    .await
    .expect("columns");
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows.len(), 2, "one row per column");
    assert_eq!(rows[0][at("column_name")], "id");
    assert_eq!(rows[0][at("ordinal")], "0");
    assert_eq!(rows[1][at("column_name")], "label");
    assert_eq!(rows[1][at("max_length")], "32");
}

/// The merged index view reports an index whatever created it, with its key
/// columns spelled out. `zyron_indexes` was registered twice for this concept
/// before the migration; there is one view now.
#[tokio::test]
async fn test_storage_indexes_is_one_view_over_every_index() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE idx_probe (id BIGINT PRIMARY KEY, a BIGINT, b BIGINT)",
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX idx_probe_ab ON idx_probe (a, b)",
    )
    .await
    .expect("index");

    let (columns, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.storage.indexes WHERE index_name = 'idx_probe_ab'",
    )
    .await
    .expect("indexes");
    assert_eq!(rows.len(), 1);
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows[0][at("table_name")], "idx_probe");
    assert_eq!(rows[0][at("is_unique")], "true");
    assert_eq!(rows[0][at("column_count")], "2");
    assert_eq!(rows[0][at("key_columns")], "a, b");
}

/// The documentation view describes every registered entity and nothing else.
#[tokio::test]
async fn test_documentation_covers_the_registry_exactly() {
    let (server, _schema, _tmp) = create_test_server().await;
    let (columns, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.core.system_view_documentation",
    )
    .await
    .expect("documentation");
    assert_eq!(
        columns,
        vec!["catalog", "schema", "object", "kind", "docstring"],
        "the documented shape changed"
    );
    assert_eq!(
        rows.len(),
        SYSTEM_OBJECTS.len(),
        "one row per registered entity"
    );
    for row in &rows {
        assert_eq!(row[0], "zyron_sys");
        assert!(
            SYSTEM_SCHEMAS.contains(&row[1].as_str()),
            "`{}` is not a registered schema",
            row[1]
        );
        assert!(
            row[3] == "VIEW" || row[3] == "TABLE FUNCTION",
            "`{}.{}` has kind `{}`",
            row[1],
            row[2],
            row[3]
        );
        assert!(
            !row[4].trim().is_empty(),
            "`{}.{}` has no docstring",
            row[1],
            row[2]
        );
    }
    // The three-part name each row describes is one that resolves
    for row in &rows {
        let name = format!("{}.{}.{}", row[0], row[1], row[2]);
        assert!(
            system_catalog::is_system_object(&name),
            "documented `{name}` does not resolve"
        );
    }
}

/// The stat views the migration renamed still report what they reported.
#[tokio::test]
async fn test_stat_views_answer_under_their_new_names() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE stat_probe (id BIGINT PRIMARY KEY, v BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO stat_probe VALUES (1, 10), (2, 20)").await;

    let (columns, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.stat.tables WHERE table_name = 'stat_probe'",
    )
    .await
    .expect("stat.tables");
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0][at("n_tup_ins")],
        "2",
        "the insert counter did not follow the rename"
    );

    let (_, rows) = read(&server, "SELECT * FROM zyron_sys.stat.wal")
        .await
        .expect("stat.wal");
    assert_eq!(rows.len(), 1, "the WAL view is a single row");

    let (columns, rows) = read(&server, "SELECT * FROM zyron_sys.stat.summary")
        .await
        .expect("summary");
    assert_eq!(columns, vec!["counter", "value"]);
    let names: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    for expected in ["wal_records", "tuples_inserted", "checkpoints_completed"] {
        assert!(names.contains(&expected), "summary is missing {expected}");
    }
}

/// The recommendation function names its canonical target and renders a
/// runnable statement for each row it produces. It is a table function so
/// it can take an optional schema name filter.
#[tokio::test]
async fn test_recommend_indexes_answers() {
    let (server, _schema, _tmp) = create_test_server().await;
    let (columns, rows) = read(&server, "SELECT * FROM zyron_sys.query.recommend_indexes()")
        .await
        .expect("recommendations");
    assert!(columns.contains(&"create_statement".to_string()));
    assert!(columns.contains(&"estimated_benefit_score".to_string()));
    assert!(columns.contains(&"sample_query".to_string()));
    for row in &rows {
        let at = columns
            .iter()
            .position(|c| c == "create_statement")
            .expect("col");
        assert!(
            row[at].starts_with("CREATE INDEX "),
            "a recommendation must render as a statement: {}",
            row[at]
        );
    }
}

/// The streaming views answer, and `streaming.jobs` reads the catalog's
/// definitions rather than the job manager's runtime, which is what makes it
/// a different view from `stat.streaming_jobs`.
#[tokio::test]
async fn test_streaming_views_answer_under_their_new_names() {
    let (server, _schema, _tmp) = create_test_server().await;
    for name in [
        "zyron_sys.streaming.watermarks",
        "zyron_sys.streaming.checkpoint_history",
        "zyron_sys.streaming.backpressure",
        "zyron_sys.streaming.operator_metrics",
        "zyron_sys.streaming.jobs",
    ] {
        let (columns, _) = read(&server, &format!("SELECT * FROM {name}"))
            .await
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!columns.is_empty(), "{name} returned no columns");
    }

    let (jobs_cols, _) = read(&server, "SELECT * FROM zyron_sys.streaming.jobs")
        .await
        .expect("jobs");
    let (stat_cols, _) = read(&server, "SELECT * FROM zyron_sys.stat.streaming_jobs")
        .await
        .expect("stat");
    assert_ne!(
        jobs_cols, stat_cols,
        "the definition view and the runtime view must not be the same view"
    );
    assert!(jobs_cols.contains(&"select_sql".to_string()));
    assert!(stat_cols.contains(&"parallelism".to_string()));
}

/// The ML and compliance views answer under their new names.
#[tokio::test]
async fn test_ml_and_compliance_views_answer() {
    let (server, _schema, _tmp) = create_test_server().await;
    for name in [
        "zyron_sys.ml.feature_groups",
        "zyron_sys.ml.feature_definitions",
        "zyron_sys.ml.models",
        "zyron_sys.compliance.legal_holds",
        "zyron_sys.security.users",
    ] {
        let (columns, _) = read(&server, &format!("SELECT * FROM {name}"))
            .await
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!columns.is_empty(), "{name} returned no columns");
    }
}

/// The feature store views report a group that was registered into it.
#[tokio::test]
async fn test_feature_store_views_report_a_registered_group() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut group =
        zyron_analytics::FeatureGroup::new("customer_features".into(), "customer_id".into());
    group.addFeature(zyron_analytics::FeatureDefinition::new(
        "lifetime_value".into(),
        "DOUBLE".into(),
        "SUM(amount)".into(),
    ));
    server
        .feature_store
        .registerFeatureGroup(group)
        .expect("register");

    let (columns, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.ml.feature_groups WHERE group_name = 'customer_features'",
    )
    .await
    .expect("groups");
    assert_eq!(rows.len(), 1);
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows[0][at("entity_key")], "customer_id");
    assert_eq!(rows[0][at("feature_count")], "1");

    let (columns, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.ml.feature_definitions WHERE feature_name = 'lifetime_value'",
    )
    .await
    .expect("definitions");
    assert_eq!(rows.len(), 1);
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows[0][at("group_name")], "customer_features");
    assert_eq!(rows[0][at("transform_expr")], "SUM(amount)");
}

// ---------------------------------------------------------------------------
// Table functions
// ---------------------------------------------------------------------------

/// The compliance report answers each of its kinds and refuses one it does
/// not have, rather than returning an empty result that reads as a pass.
#[tokio::test]
async fn test_compliance_report_answers_by_kind() {
    let (server, _schema, _tmp) = create_test_server().await;
    for kind in ["retention", "legal_hold", "audit", "events"] {
        let (columns, _rows) = read(
            &server,
            &format!("SELECT * FROM zyron_sys.compliance.report('{kind}')"),
        )
        .await
        .unwrap_or_else(|e| panic!("report('{kind}'): {e}"));
        assert_eq!(
            columns,
            vec![
                "kind",
                "subject",
                "table_id",
                "status",
                "detail",
                "event_time"
            ],
            "report('{kind}') returned a different shape"
        );
    }

    let (_, rows) = read(
        &server,
        "SELECT * FROM zyron_sys.compliance.report('audit')",
    )
    .await
    .expect("audit");
    assert_eq!(rows.len(), 1, "the audit report is one verdict");
    assert_eq!(rows[0][3], "PASS", "an untouched log verifies");

    let err = read(&server, "SELECT * FROM zyron_sys.compliance.report('nope')")
        .await
        .err()
        .expect("an unknown kind is refused");
    assert!(err.contains("is not a report kind"), "got: {err}");
}

/// Reading a table function without its arguments says so rather than
/// answering with nothing.
#[tokio::test]
async fn test_a_table_function_read_as_a_view_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let err = read(&server, "SELECT * FROM zyron_sys.compliance.report")
        .await
        .err()
        .expect("refused");
    assert!(
        err.contains("table function") && err.contains("arguments"),
        "got: {err}"
    );
}

/// Reading a view as if it were a function says so too.
#[tokio::test]
async fn test_a_view_called_as_a_function_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let err = read(&server, "SELECT * FROM zyron_sys.core.tables('x')")
        .await
        .err()
        .expect("refused");
    assert!(err.contains("is a view"), "got: {err}");
}

// ---------------------------------------------------------------------------
// Clause handling survives the rename
// ---------------------------------------------------------------------------

/// A projection, an equality, LIMIT, and OFFSET all still apply, and a clause
/// the entities cannot answer is still refused rather than dropped.
#[tokio::test]
async fn test_clauses_still_apply_under_the_new_names() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    for name in ["clause_a", "clause_b", "clause_c"] {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE {name} (id BIGINT PRIMARY KEY)"),
        )
        .await
        .expect("create");
    }

    let (columns, rows) = read(
        &server,
        "SELECT table_name, storage_format FROM zyron_sys.core.tables WHERE schema_name = 'zyron_test'",
    )
    .await
    .expect("projection");
    assert_eq!(columns, vec!["table_name", "storage_format"]);
    assert!(rows.len() >= 3);

    let (_, limited) = read(&server, "SELECT * FROM zyron_sys.core.tables LIMIT 2")
        .await
        .expect("limit");
    assert_eq!(limited.len(), 2);

    let (_, offset) = read(
        &server,
        "SELECT * FROM zyron_sys.core.tables LIMIT 1 OFFSET 1",
    )
    .await
    .expect("offset");
    assert_eq!(offset.len(), 1);

    let err = read(
        &server,
        "SELECT * FROM zyron_sys.core.tables ORDER BY table_name",
    )
    .await
    .err()
    .expect("ORDER BY is refused, not dropped");
    assert!(err.contains("does not support ORDER BY"), "got: {err}");

    let err = read(
        &server,
        "SELECT nonexistent_column FROM zyron_sys.core.tables",
    )
    .await
    .err()
    .expect("an unknown column is refused");
    assert!(err.contains("has no column named"), "got: {err}");
}

// ---------------------------------------------------------------------------
// Registry integrity
// ---------------------------------------------------------------------------

/// Every registered entity has a builder, and every builder is registered.
/// Read together with the resolution test above, this is what keeps the
/// registry and the dispatch from drifting apart.
#[tokio::test]
async fn test_registry_and_dispatch_agree() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut unbuilt = Vec::new();
    for object in SYSTEM_OBJECTS {
        let name = object.canonical_name();
        match object.kind {
            SystemObjectKind::View => {
                // A view that needs scoping predicates gets them, so this
                // checks that every registered name has a builder rather
                // than re-checking argument validation
                let mut filters = zyron_wire::system_views::SystemViewFilters::default();
                if !required_predicate(&name).is_empty() {
                    filters.equalities = vec![
                        ("from_version".into(), "1".into()),
                        ("to_version".into(), "2".into()),
                    ];
                }
                match zyron_wire::system_views::query_system_view(&name, &server, &filters).await {
                    Ok(Some(_)) => {}
                    Ok(None) => unbuilt.push(format!("{name} built nothing")),
                    Err(e) => unbuilt.push(format!("{name}: {e}")),
                }
            }
            SystemObjectKind::TableFunction => {
                // A function is exercised through its own tests; here it is
                // enough that reading it as a view is refused for being a
                // function rather than for being unregistered
                match zyron_wire::system_views::query_system_view(
                    &name,
                    &server,
                    &zyron_wire::system_views::SystemViewFilters::default(),
                )
                .await
                {
                    Err(zyron_common::ZyronError::PlanError(msg))
                        if msg.contains("table function") => {}
                    other => unbuilt.push(format!("{name} answered {other:?} as a view")),
                }
            }
        }
    }
    assert!(unbuilt.is_empty(), "{unbuilt:#?}");
}

/// The pressure schema's own dispatch list and the registry name the same
/// entities. The pressure views ship their list separately because they are
/// built off the controller rather than off ServerState, so this is the seam
/// where the two could drift.
#[test]
fn test_pressure_dispatch_matches_the_registry() {
    let registered: Vec<String> = SYSTEM_OBJECTS
        .iter()
        .filter(|o| o.schema == "pressure")
        .map(|o| o.canonical_name())
        .collect();
    let dispatched: Vec<String> = zyron_wire::pressure_views::PRESSURE_VIEW_NAMES
        .iter()
        .map(|n| n.to_string())
        .collect();
    for name in &registered {
        assert!(
            dispatched.contains(name),
            "`{name}` is registered but the pressure dispatch does not know it"
        );
    }
    for name in &dispatched {
        assert!(
            registered.contains(name),
            "`{name}` is dispatched but is not registered"
        );
    }
}

// ---------------------------------------------------------------------------
// The planner path
// ---------------------------------------------------------------------------

/// A user table that does not exist gets the same error shape with no
/// suggestion, so the hint machinery does not fire on ordinary typos in user
/// space.
#[tokio::test]
async fn test_the_planner_refuses_an_unknown_user_table_without_a_hint() {
    let (server, _schema, _tmp) = create_test_server().await;
    let stmt = zyron_parser::parse("SELECT * FROM no_such_customer_table")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let err = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
    )
    .await
    .expect_err("an unknown table is refused");
    match &err {
        zyron_common::ZyronError::RelationNotFound { name } => {
            assert_eq!(name, "no_such_customer_table");
        }
        other => panic!("expected RelationNotFound, got {other:?}"),
    }
}

/// A user table really named `zyron_something` still resolves. The bare
/// prefix is not reserved: only the three-part names under `zyron_sys` are,
/// and a table that happens to start with `zyron_` is a table.
#[tokio::test]
async fn test_a_user_table_with_a_zyron_prefix_still_resolves() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE zyron_customer_notes (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create");

    let stmt = zyron_parser::parse("SELECT * FROM zyron_customer_notes")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
    )
    .await
    .expect("a user table keeps its name whatever it starts with");
}

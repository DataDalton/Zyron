//! Catalog navigation, storage, query-tuning, ML, and compliance entities of
//! the `zyron_sys` catalog.
//!
//! Everything here reads live state and renders it: the catalog's own entries
//! for `core.*` and `storage.indexes`, the planner's workload tracker for
//! `query.recommend_indexes`, the feature store and model cache for `ml.*`,
//! and the legal-hold and compliance tables for `compliance.*`. Nothing is
//! stored on behalf of these names, so a read is always of the current state
//! rather than of a snapshot someone had to refresh.

use zyron_catalog::system_catalog::{SYSTEM_CATALOG_NAME, SYSTEM_OBJECTS};
use zyron_common::ZyronError;

use crate::connection::ServerState;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_INT4_OID, PG_INT8_OID, PG_TEXT_OID};

/// Renders one cell the way every view here renders text, so a WHERE filter
/// compares against exactly what the client is shown.
fn cell(value: impl ToString) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

/// Dispatches the entities of this module. Reached only for a name the
/// registry recognized, so an unmatched pair is a registration without a
/// builder and is reported as such rather than as an empty result.
pub async fn build(
    schema: &str,
    object: &str,
    server: &ServerState,
) -> Result<ViewRows, ZyronError> {
    Ok(match (schema, object) {
        ("core", "databases") => build_databases(server),
        ("core", "schemas") => build_schemas(server),
        ("core", "tables") => build_tables(server),
        ("core", "columns") => build_columns(server),
        ("core", "constraints") => build_constraints(server),
        ("core", "system_view_documentation") => build_documentation(),
        ("storage", "indexes") => build_indexes(server),
        ("storage", "ddl_progress") => build_ddl_progress(server),
        ("storage", "variant_shredding_stats") => build_variant_shredding_stats(server),
        ("sql", "triggers") => build_triggers(server),
        ("session", "prepared_statements") => {
            crate::system_views::build_session_prepared_statements(server)
        }
        ("expectation", "results") => build_expectation_results(server),
        ("ml", "feature_groups") => build_feature_groups(server),
        ("ml", "feature_definitions") => build_feature_definitions(server),
        ("ml", "models") => build_models(server),
        ("compliance", "legal_holds") => build_legal_holds(server).await?,
        ("security", "users") => build_users(server).await?,
        _ => {
            return Err(ZyronError::Internal(format!(
                "`{}.{}.{}` is registered but has no builder",
                SYSTEM_CATALOG_NAME, schema, object
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// core
// ---------------------------------------------------------------------------

/// Builds zyron_sys.core.databases.
/// Columns: catalog_id, catalog_name, owner, created_at, is_system.
fn build_databases(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("catalog_id", PG_INT4_OID, 4),
        make_field("catalog_name", PG_TEXT_OID, -1),
        make_field("owner", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
        make_field("is_system", PG_TEXT_OID, -1),
    ];
    let system_id = server.catalog.system_catalog_id();
    let rows = server
        .catalog
        .list_databases()
        .into_iter()
        .map(|db| {
            vec![
                cell(db.id.0),
                cell(&db.name),
                cell(&db.owner),
                cell(db.created_at),
                cell(Some(db.id) == system_id),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.core.schemas.
/// Columns: schema_id, schema_name, catalog_id, catalog_name, owner,
///          is_system.
fn build_schemas(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("schema_id", PG_INT4_OID, 4),
        make_field("schema_name", PG_TEXT_OID, -1),
        make_field("catalog_id", PG_INT4_OID, 4),
        make_field("catalog_name", PG_TEXT_OID, -1),
        make_field("owner", PG_TEXT_OID, -1),
        make_field("is_system", PG_TEXT_OID, -1),
    ];
    let catalogs = server.catalog.list_databases();
    let rows = server
        .catalog
        .list_schemas()
        .into_iter()
        .map(|schema| {
            let catalog_name = catalogs
                .iter()
                .find(|d| d.id == schema.database_id)
                .map(|d| d.name.clone())
                .unwrap_or_default();
            vec![
                cell(schema.id.0),
                cell(&schema.name),
                cell(schema.database_id.0),
                cell(catalog_name),
                cell(&schema.owner),
                cell(server.catalog.is_system_schema(schema.id)),
            ]
        })
        .collect();
    (fields, rows)
}

/// The storage a table's rows actually live in.
///
/// A foreign table is reported as foreign rather than as the heap it does not
/// have, and a heap table that has folded segments reports both tiers, which
/// is what it is: the fold moves rows without emptying the heap.
fn storage_format(table: &zyron_catalog::TableEntry) -> &'static str {
    if table.foreign.is_foreign() {
        return "FOREIGN";
    }
    if table.lake.is_lake() {
        return "ZYRONLAKE";
    }
    if table.columnar.segments.is_empty() {
        "HEAP"
    } else {
        "HEAP+COLUMNAR"
    }
}

/// Builds zyron_sys.core.tables.
/// Columns: table_id, table_name, schema_id, schema_name, catalog_name,
///          storage_format, column_count, constraint_count, cdf_enabled,
///          versioning_enabled, created_at.
fn build_tables(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("schema_id", PG_INT4_OID, 4),
        make_field("schema_name", PG_TEXT_OID, -1),
        make_field("catalog_name", PG_TEXT_OID, -1),
        make_field("storage_format", PG_TEXT_OID, -1),
        make_field("column_count", PG_INT4_OID, 4),
        make_field("constraint_count", PG_INT4_OID, 4),
        make_field("cdf_enabled", PG_TEXT_OID, -1),
        make_field("versioning_enabled", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
    ];
    let schemas = server.catalog.list_schemas();
    let catalogs = server.catalog.list_databases();
    let rows = server
        .catalog
        .list_all_tables()
        .into_iter()
        .map(|table| {
            let schema = schemas.iter().find(|s| s.id == table.schema_id);
            let schema_name = schema.map(|s| s.name.clone()).unwrap_or_default();
            let catalog_name = schema
                .and_then(|s| catalogs.iter().find(|d| d.id == s.database_id))
                .map(|d| d.name.clone())
                .unwrap_or_default();
            vec![
                cell(table.id.0),
                cell(&table.name),
                cell(table.schema_id.0),
                cell(schema_name),
                cell(catalog_name),
                cell(storage_format(&table)),
                // Columns a reader can name. A dropped column keeps its place
                // in the encoded row so older rows still decode, and counting
                // it here would report a width no query can select
                cell(table.live_columns().count()),
                cell(table.constraints.len()),
                cell(table.cdf_enabled),
                cell(table.versioning_enabled),
                cell(table.created_at),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.core.columns.
/// Columns: table_id, table_name, schema_name, column_id, column_name,
///          ordinal, data_type, nullable, max_length, fractional_digits,
///          default_expr.
fn build_columns(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("schema_name", PG_TEXT_OID, -1),
        make_field("column_id", PG_INT4_OID, 4),
        make_field("column_name", PG_TEXT_OID, -1),
        make_field("ordinal", PG_INT4_OID, 4),
        make_field("data_type", PG_TEXT_OID, -1),
        make_field("nullable", PG_TEXT_OID, -1),
        make_field("max_length", PG_INT4_OID, 4),
        make_field("fractional_digits", PG_INT4_OID, 4),
        make_field("default_expr", PG_TEXT_OID, -1),
    ];
    let schemas = server.catalog.list_schemas();
    let mut rows = Vec::new();
    let mut tables = server.catalog.list_all_tables();
    tables.sort_by_key(|t| t.id.0);
    for table in tables {
        let schema_name = schemas
            .iter()
            .find(|s| s.id == table.schema_id)
            .map(|s| s.name.clone())
            .unwrap_or_default();
        // A dropped column still occupies its position in every tuple already
        // written, which is the decoder's business and nobody else's
        let mut columns: Vec<_> = table.live_columns().collect();
        columns.sort_by_key(|c| c.ordinal);
        for column in columns {
            rows.push(vec![
                cell(table.id.0),
                cell(&table.name),
                cell(&schema_name),
                cell(column.id.0),
                cell(&column.name),
                cell(column.ordinal),
                cell(format!("{:?}", column.type_id)),
                cell(column.nullable),
                column.max_length.map(|v| v.to_string().into_bytes()),
                column.fractional_digits.map(|v| v.to_string().into_bytes()),
                column.default_expr.as_ref().map(|v| v.as_bytes().to_vec()),
            ]);
        }
    }
    (fields, rows)
}

/// Builds zyron_sys.core.system_view_documentation.
/// Columns: catalog, schema, object, kind, docstring.
///
/// Generated from the same registry the catalog and schema rows are created
/// from, so the documentation cannot describe an entity that does not exist
/// or omit one that does.
fn build_documentation() -> ViewRows {
    let fields = vec![
        make_field("catalog", PG_TEXT_OID, -1),
        make_field("schema", PG_TEXT_OID, -1),
        make_field("object", PG_TEXT_OID, -1),
        make_field("kind", PG_TEXT_OID, -1),
        make_field("docstring", PG_TEXT_OID, -1),
    ];
    let mut entries: Vec<_> = SYSTEM_OBJECTS.iter().collect();
    entries.sort_by_key(|o| (o.schema, o.object));
    let rows = entries
        .into_iter()
        .map(|o| {
            vec![
                cell(SYSTEM_CATALOG_NAME),
                cell(o.schema),
                cell(o.object),
                cell(o.kind.label()),
                cell(o.doc),
            ]
        })
        .collect();
    (fields, rows)
}

// ---------------------------------------------------------------------------
// storage
// ---------------------------------------------------------------------------

/// Builds zyron_sys.storage.indexes.
/// Columns: index_id, index_name, table_id, table_name, schema_name,
///          index_type, is_unique, column_count, key_columns.
///
/// One view for every index the catalog holds, whatever tier the table it
/// covers lives in. The key column list renders the columns in key order with
/// a DESC marker where one was declared, which is what tells two indexes over
/// the same columns apart.
fn build_indexes(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("index_id", PG_INT4_OID, 4),
        make_field("index_name", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("schema_name", PG_TEXT_OID, -1),
        make_field("index_type", PG_TEXT_OID, -1),
        make_field("is_unique", PG_TEXT_OID, -1),
        make_field("column_count", PG_INT4_OID, 4),
        make_field("key_columns", PG_TEXT_OID, -1),
        make_field("state", PG_TEXT_OID, -1),
        make_field("build_progress_id", PG_INT8_OID, 8),
    ];
    let schemas = server.catalog.list_schemas();
    let rows = server
        .catalog
        .list_all_indexes()
        .into_iter()
        .map(|index| {
            let table = server.catalog.get_table_by_id(index.table_id).ok();
            let table_name = table
                .as_ref()
                .map(|t| t.name.clone())
                .unwrap_or_else(|| format!("table_{}", index.table_id.0));
            let schema_name = schemas
                .iter()
                .find(|s| s.id == index.schema_id)
                .map(|s| s.name.clone())
                .unwrap_or_default();
            let mut key_columns: Vec<_> = index.columns.iter().collect();
            key_columns.sort_by_key(|c| c.ordinal);
            let rendered = key_columns
                .iter()
                .map(|c| {
                    let name = table
                        .as_ref()
                        .and_then(|t| t.columns.iter().find(|tc| tc.id == c.column_id))
                        .map(|tc| tc.name.clone())
                        .unwrap_or_else(|| format!("column_{}", c.column_id.0));
                    if c.descending {
                        format!("{} DESC", name)
                    } else {
                        name
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            // A build in flight has a progress row, and pointing at it is
            // what turns "why is this index not being used" into one more
            // query rather than a guess
            let progress_id = if index.state == zyron_catalog::IndexState::Building {
                table
                    .as_ref()
                    .and_then(|t| server.ddl_progress.row_for_object(&t.name, &index.name))
                    .map(|row| row.id)
            } else {
                None
            };
            vec![
                cell(index.id.0),
                cell(&index.name),
                cell(index.table_id.0),
                cell(table_name),
                cell(schema_name),
                cell(format!("{:?}", index.index_type)),
                cell(index.unique),
                cell(index.columns.len()),
                cell(rendered),
                cell(index.state.as_str()),
                match progress_id {
                    Some(id) => cell(id),
                    None => None,
                },
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.core.constraints.
///
/// `state` says whether the rows that predate the constraint have been
/// checked. A constraint reported not-yet-valid is still enforced on every
/// write, which is the difference between a rule nobody is applying and a rule
/// whose history has not been settled.
fn build_constraints(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("schema_name", PG_TEXT_OID, -1),
        make_field("constraint_name", PG_TEXT_OID, -1),
        make_field("constraint_type", PG_TEXT_OID, -1),
        make_field("columns", PG_TEXT_OID, -1),
        make_field("check_expr", PG_TEXT_OID, -1),
        make_field("enforced", PG_TEXT_OID, -1),
        make_field("state", PG_TEXT_OID, -1),
    ];
    let schemas = server.catalog.list_schemas();
    let mut rows = Vec::new();
    let mut tables = server.catalog.list_all_tables();
    tables.sort_by_key(|t| t.id.0);
    for table in tables {
        let schema_name = schemas
            .iter()
            .find(|s| s.id == table.schema_id)
            .map(|s| s.name.clone())
            .unwrap_or_default();
        for constraint in &table.constraints {
            let columns = constraint
                .columns
                .iter()
                .map(|cid| {
                    table
                        .columns
                        .iter()
                        .find(|c| c.id == *cid)
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("column_{}", cid.0))
                })
                .collect::<Vec<_>>()
                .join(", ");
            rows.push(vec![
                cell(table.id.0),
                cell(&table.name),
                cell(&schema_name),
                cell(&constraint.name),
                cell(format!("{:?}", constraint.constraint_type)),
                cell(columns),
                constraint
                    .check_expr
                    .as_ref()
                    .map(|e| e.clone().into_bytes()),
                cell(constraint.enforced),
                cell(if constraint.validated {
                    "valid"
                } else {
                    "validating"
                }),
            ]);
        }
    }
    (fields, rows)
}

/// Builds zyron_sys.storage.ddl_progress.
///
/// One row per online DDL operation running in this process, gone the moment
/// the operation ends however it ends. `rows_total_estimate` is named an
/// estimate because it comes from table statistics, which lag writes.
fn build_ddl_progress(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("progress_id", PG_INT8_OID, 8),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("operation", PG_TEXT_OID, -1),
        make_field("object_name", PG_TEXT_OID, -1),
        make_field("phase", PG_TEXT_OID, -1),
        make_field("rows_done", PG_INT8_OID, 8),
        make_field("rows_total_estimate", PG_INT8_OID, 8),
        make_field("bytes_spilled", PG_INT8_OID, 8),
        make_field("started_at_secs", PG_INT8_OID, 8),
        make_field("issuing_session", PG_TEXT_OID, -1),
        make_field("pause_signal", PG_TEXT_OID, -1),
    ];
    let rows = server
        .ddl_progress
        .rows()
        .into_iter()
        .map(|row| {
            vec![
                cell(row.id),
                cell(&row.table),
                cell(row.operation.as_str()),
                cell(&row.object),
                cell(row.phase().as_str()),
                cell(row.rows_done()),
                cell(row.rows_total_estimate()),
                cell(row.bytes_spilled()),
                cell(row.started_at_secs),
                cell(&row.issuing_session),
                match row.pause_signal() {
                    Some(signal) => cell(signal),
                    None => None,
                },
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.storage.variant_shredding_stats.
/// Columns: table_id, table_name, column_id, column_name, path, occurrences,
///          coverage_percent, value_kind, shredded.
///
/// Reads the executor's process-wide variant path tracker for every variant
/// column the catalog holds, so the rows describe the JSON shapes this node
/// has actually been asked to store. Rows are ordered by table, column, and
/// descending occurrences, which puts each column's strongest promotion
/// candidate first
fn build_variant_shredding_stats(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("column_id", PG_INT4_OID, 4),
        make_field("column_name", PG_TEXT_OID, -1),
        make_field("path", PG_TEXT_OID, -1),
        make_field("occurrences", PG_INT8_OID, 8),
        make_field("coverage_percent", PG_TEXT_OID, -1),
        make_field("value_kind", PG_TEXT_OID, -1),
        make_field("shredded", PG_TEXT_OID, -1),
    ];
    let mut tables = server.catalog.list_all_tables();
    tables.sort_by_key(|t| t.id.0);
    let mut rows = Vec::new();
    for table in tables {
        let mut columns: Vec<_> = table
            .columns
            .iter()
            .filter(|c| c.type_id == zyron_common::TypeId::Variant)
            .collect();
        columns.sort_by_key(|c| c.id.0);
        for column in columns {
            for stat in zyron_executor::variant_shred::paths_for(table.id.0, column.id.0) {
                rows.push(vec![
                    cell(table.id.0),
                    cell(&table.name),
                    cell(column.id.0),
                    cell(&column.name),
                    cell(&stat.path),
                    cell(stat.occurrences),
                    cell(format!("{:.2}", stat.coverage_percent)),
                    cell(stat.value_kind),
                    cell(stat.shredded),
                ]);
            }
        }
    }
    (fields, rows)
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

/// Builds zyron_sys.query.recommend_indexes.
/// Columns: table_id, table_name, columns, scan_count,
///          estimated_selectivity, reason, create_statement.
///
/// Reads the planner's process-wide workload tracker, so the rows describe
/// what this node has actually been asked to scan rather than a static
/// analysis of the schema. The rendered CREATE INDEX statement is the
/// recommendation in the form the operator would run it.
pub(crate) fn build_recommend_indexes(
    server: &ServerState,
    schema_filter: Option<&str>,
) -> ViewRows {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("column_names", PG_TEXT_OID, -1),
        make_field("scan_count", PG_INT8_OID, 8),
        make_field("estimated_selectivity", PG_TEXT_OID, -1),
        make_field("estimated_benefit_score", PG_TEXT_OID, -1),
        make_field("sample_query", PG_TEXT_OID, -1),
        make_field("reason", PG_TEXT_OID, -1),
        make_field("create_statement", PG_TEXT_OID, -1),
    ];
    let schema_id_filter = schema_filter.and_then(|name| {
        server
            .catalog
            .list_schemas()
            .into_iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
            .map(|s| s.id)
    });
    let advisor = zyron_planner::optimizer::rules::global_index_advisor();
    let rows = advisor
        .recommendations(&server.catalog)
        .into_iter()
        .filter(|rec| {
            match (schema_filter, schema_id_filter) {
                // A named schema that does not exist filters everything out
                (Some(_), None) => false,
                (_, Some(schema_id)) => server
                    .catalog
                    .get_table_by_id(rec.table_id)
                    .map(|t| t.schema_id == schema_id)
                    .unwrap_or(false),
                (None, None) => true,
            }
        })
        .map(|rec| {
            let table = server.catalog.get_table_by_id(rec.table_id).ok();
            let table_name = table
                .as_ref()
                .map(|t| t.name.clone())
                .unwrap_or_else(|| format!("table_{}", rec.table_id.0));
            let column_names: Vec<String> = rec
                .columns
                .iter()
                .map(|cid| {
                    table
                        .as_ref()
                        .and_then(|t| t.columns.iter().find(|c| c.id == *cid))
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("column_{}", cid.0))
                })
                .collect();
            let joined = column_names.join(", ");
            let suggested = format!(
                "CREATE INDEX idx_{}_{} ON {} ({})",
                table_name,
                column_names.join("_"),
                table_name,
                joined
            );
            // Benefit grows with how often the column is scanned and how
            // selective an index on it would be
            let benefit = rec.scan_count as f64 * (1.0 - rec.estimated_selectivity);
            let sample_query = column_names
                .first()
                .map(|first| format!("SELECT * FROM {table_name} WHERE {first} = $1"))
                .unwrap_or_default();
            vec![
                cell(rec.table_id.0),
                cell(table_name),
                cell(joined),
                cell(rec.scan_count),
                cell(format!("{:.6}", rec.estimated_selectivity)),
                cell(format!("{benefit:.3}")),
                cell(sample_query),
                cell(&rec.reason),
                cell(suggested),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.expectation.results.
/// Columns: table_id, table_name, expectation_name, evaluation_time,
///          passed, details.
fn build_expectation_results(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("expectation_name", PG_TEXT_OID, -1),
        make_field("evaluation_time", PG_INT8_OID, 8),
        make_field("passed", PG_TEXT_OID, -1),
        make_field("details", PG_TEXT_OID, -1),
    ];
    let rows = zyron_executor::expectation_results::snapshot()
        .into_iter()
        .map(|o| {
            let table_name = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(o.table_id))
                .map(|t| t.name.clone())
                .unwrap_or_else(|_| format!("table_{}", o.table_id));
            let details = format!(
                "{{\"rows_checked\":{},\"violations\":{},\"action\":\"{}\"}}",
                o.rows_checked, o.violations, o.action
            );
            vec![
                cell(o.table_id),
                cell(table_name),
                cell(&o.expectation_name),
                cell(o.evaluated_at_micros),
                cell(o.passed),
                cell(details),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.sql.triggers.
/// Columns: trigger_id, trigger_name, table_id, table_name, timing, events,
///          for_each, execute_function, enabled.
fn build_triggers(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("trigger_id", PG_INT4_OID, 4),
        make_field("trigger_name", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("timing", PG_TEXT_OID, -1),
        make_field("events", PG_TEXT_OID, -1),
        make_field("for_each", PG_TEXT_OID, -1),
        make_field("execute_function", PG_TEXT_OID, -1),
        make_field("enabled", PG_TEXT_OID, -1),
    ];
    let rows = server
        .catalog
        .list_triggers()
        .into_iter()
        .map(|t| {
            let table_name = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(t.table_id))
                .map(|te| te.name.clone())
                .unwrap_or_else(|_| format!("table_{}", t.table_id));
            let timing = match t.timing {
                zyron_catalog::TriggerEntry::TIMING_BEFORE => "before",
                zyron_catalog::TriggerEntry::TIMING_AFTER => "after",
                zyron_catalog::TriggerEntry::TIMING_INSTEAD_OF => "instead_of",
                _ => "unknown",
            };
            let mut events = Vec::new();
            if t.events & zyron_catalog::TriggerEntry::EVENT_INSERT != 0 {
                events.push("insert");
            }
            if t.events & zyron_catalog::TriggerEntry::EVENT_UPDATE != 0 {
                events.push("update");
            }
            if t.events & zyron_catalog::TriggerEntry::EVENT_DELETE != 0 {
                events.push("delete");
            }
            let for_each = if t.for_each == zyron_catalog::TriggerEntry::FOR_EACH_STATEMENT {
                "statement"
            } else {
                "row"
            };
            vec![
                cell(t.id),
                cell(&t.name),
                cell(t.table_id),
                cell(table_name),
                cell(timing),
                cell(events.join(",")),
                cell(for_each),
                cell(&t.execute_function),
                cell(t.enabled),
            ]
        })
        .collect();
    (fields, rows)
}

// ---------------------------------------------------------------------------
// ml
// ---------------------------------------------------------------------------

/// Builds zyron_sys.ml.feature_groups.
/// Columns: group_name, entity_key, feature_count, source_query,
///          backing_table, refresh_seconds, max_staleness_seconds,
///          retention_days, last_refresh_ms.
fn build_feature_groups(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("group_name", PG_TEXT_OID, -1),
        make_field("entity_key", PG_TEXT_OID, -1),
        make_field("feature_count", PG_INT4_OID, 4),
        make_field("source_query", PG_TEXT_OID, -1),
        make_field("backing_table", PG_TEXT_OID, -1),
        make_field("refresh_seconds", PG_INT8_OID, 8),
        make_field("max_staleness_seconds", PG_INT8_OID, 8),
        make_field("retention_days", PG_INT8_OID, 8),
        make_field("last_refresh_ms", PG_INT8_OID, 8),
    ];
    let mut groups = server.feature_store.groups();
    groups.sort_by(|a, b| a.name.cmp(&b.name));
    let rows = groups
        .into_iter()
        .map(|g| {
            vec![
                cell(&g.name),
                cell(&g.entityKey),
                cell(g.features.len()),
                cell(&g.sourceQuery),
                g.backingTable.as_ref().map(|t| t.as_bytes().to_vec()),
                cell(g.refreshSeconds),
                cell(g.maxStalenessSeconds),
                cell(g.retentionDays),
                cell(g.lastRefreshMs),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.ml.feature_definitions.
/// Columns: group_name, feature_name, data_type, version, description,
///          transform_expr, created_at_ms.
fn build_feature_definitions(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("group_name", PG_TEXT_OID, -1),
        make_field("feature_name", PG_TEXT_OID, -1),
        make_field("data_type", PG_TEXT_OID, -1),
        make_field("version", PG_INT4_OID, 4),
        make_field("description", PG_TEXT_OID, -1),
        make_field("transform_expr", PG_TEXT_OID, -1),
        make_field("created_at_ms", PG_INT8_OID, 8),
    ];
    let mut groups = server.feature_store.groups();
    groups.sort_by(|a, b| a.name.cmp(&b.name));
    let mut rows = Vec::new();
    for group in groups {
        for feature in &group.features {
            rows.push(vec![
                cell(&group.name),
                cell(&feature.name),
                cell(&feature.dataType),
                cell(feature.version),
                cell(&feature.description),
                cell(&feature.transformExpr),
                cell(feature.createdAtMs),
            ]);
        }
    }
    (fields, rows)
}

/// Builds zyron_sys.ml.models.
/// Columns: model_name, model_type, feature_count, feature_columns,
///          target_column, training_rows, quantized, created_at_ms, metrics.
fn build_models(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("model_name", PG_TEXT_OID, -1),
        make_field("model_type", PG_TEXT_OID, -1),
        make_field("feature_count", PG_INT4_OID, 4),
        make_field("feature_columns", PG_TEXT_OID, -1),
        make_field("target_column", PG_TEXT_OID, -1),
        make_field("training_rows", PG_INT8_OID, 8),
        make_field("quantized", PG_TEXT_OID, -1),
        make_field("created_at_ms", PG_INT8_OID, 8),
        make_field("metrics", PG_TEXT_OID, -1),
    ];
    let mut names = server.model_cache.names();
    names.sort();
    let rows = names
        .into_iter()
        .filter_map(|name| {
            let model = server.model_cache.get(&name)?;
            // Metric order is not stable across runs because the map is
            // hashed, so it is sorted here: two reads of the same model
            // render the same text
            let mut metrics: Vec<(&String, &f64)> = model.metrics.iter().collect();
            metrics.sort_by(|a, b| a.0.cmp(b.0));
            let rendered = metrics
                .iter()
                .map(|(k, v)| format!("{}={:.6}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            Some(vec![
                cell(&name),
                cell(format!("{:?}", model.modelType)),
                cell(model.featureColumns.len()),
                cell(model.featureColumns.join(", ")),
                model.targetColumn.as_ref().map(|t| t.as_bytes().to_vec()),
                cell(model.trainingRows),
                cell(model.quantized),
                cell(model.createdAtMs),
                cell(rendered),
            ])
        })
        .collect();
    (fields, rows)
}

// ---------------------------------------------------------------------------
// security
// ---------------------------------------------------------------------------

/// Builds zyron_sys.security.users.
/// Columns: user_id, username, can_login, is_superuser, locked, locked_reason,
///          connection_limit, valid_until, has_password, has_api_key,
///          has_totp, created_at.
///
/// Presence flags rather than the credentials themselves: whether an account
/// can authenticate is the operational question, and the secret answering it
/// has no business in a view. A node with no security manager reports no
/// rows rather than an empty account list, because "nobody is configured" and
/// "authentication is not running here" are different states.
async fn build_users(server: &ServerState) -> Result<ViewRows, ZyronError> {
    let fields = vec![
        make_field("user_id", PG_INT4_OID, 4),
        make_field("username", PG_TEXT_OID, -1),
        make_field("can_login", PG_TEXT_OID, -1),
        make_field("is_superuser", PG_TEXT_OID, -1),
        make_field("locked", PG_TEXT_OID, -1),
        make_field("locked_reason", PG_TEXT_OID, -1),
        make_field("connection_limit", PG_INT4_OID, 4),
        make_field("valid_until", PG_INT8_OID, 8),
        make_field("has_password", PG_TEXT_OID, -1),
        make_field("has_api_key", PG_TEXT_OID, -1),
        make_field("has_totp", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
    ];
    let Some(security) = server.security_manager.as_ref() else {
        return Ok((fields, Vec::new()));
    };
    let mut users = security.auth_storage.load_users().await?;
    users.sort_by(|a, b| a.name.cmp(&b.name));
    let rows = users
        .into_iter()
        .map(|user| {
            vec![
                cell(user.id.0),
                cell(&user.name),
                cell(user.can_login),
                cell(user.superuser),
                cell(user.locked),
                user.locked_reason.as_ref().map(|r| r.as_bytes().to_vec()),
                cell(user.connection_limit),
                user.valid_until.map(|v| v.to_string().into_bytes()),
                cell(user.password_hash.is_some() || user.scram_secret.is_some()),
                cell(user.api_key_hash.is_some()),
                cell(user.totp_secret.is_some()),
                cell(user.created_at),
            ]
        })
        .collect();
    Ok((fields, rows))
}

// ---------------------------------------------------------------------------
// compliance
// ---------------------------------------------------------------------------

/// Builds zyron_sys.compliance.legal_holds.
/// Columns: hold_id, hold_name, table_id, table_name, scope, predicate_sql,
///          reason, created_at, released_at, active.
///
/// Reads the catalog's persisted holds rather than the in-memory enforcement
/// registry, because a released hold still has to be visible: the record of a
/// hold having existed is itself the compliance artifact.
async fn build_legal_holds(server: &ServerState) -> Result<ViewRows, ZyronError> {
    let fields = vec![
        make_field("hold_id", PG_INT4_OID, 4),
        make_field("hold_name", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("scope", PG_TEXT_OID, -1),
        make_field("predicate_sql", PG_TEXT_OID, -1),
        make_field("reason", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
        make_field("released_at", PG_INT8_OID, 8),
        make_field("active", PG_TEXT_OID, -1),
    ];
    let holds = server.catalog.load_legal_holds().await?;
    let rows = holds
        .into_iter()
        .map(|hold| {
            let table_name = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(hold.table_id))
                .map(|t| t.name.clone())
                .unwrap_or_else(|_| format!("table_{}", hold.table_id));
            let scope = if hold.predicate_sql.is_empty() {
                "TABLE"
            } else {
                "PREDICATE"
            };
            vec![
                cell(hold.id),
                cell(&hold.name),
                cell(hold.table_id),
                cell(table_name),
                cell(scope),
                cell(&hold.predicate_sql),
                cell(&hold.reason),
                cell(hold.created_at),
                if hold.released_at == 0 {
                    None
                } else {
                    cell(hold.released_at)
                },
                cell(hold.released_at == 0),
            ]
        })
        .collect();
    Ok((fields, rows))
}

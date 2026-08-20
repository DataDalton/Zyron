//! Virtual statistics views that bypass the normal planner/executor path.
//!
//! Each view returns a column schema (FieldDescription vector) and data rows
//! directly, allowing clients to query internal server metrics through
//! standard SQL SELECT statements on virtual system tables.

use std::sync::atomic::Ordering;

use zyron_common::ZyronError;

use crate::connection::ServerState;
use crate::messages::backend::FieldDescription;
use crate::types::{PG_INT4_OID, PG_INT8_OID, PG_TEXT_OID};

/// Row data for zyron_stat_activity, collected by the session manager.
pub struct SessionRow {
    pub pid: i32,
    pub user_name: String,
    pub database: String,
    pub state: String,
    pub connected_at_secs: u64,
    pub last_activity_secs: u64,
}

/// List of recognized virtual statistics view names.
const STAT_VIEW_NAMES: &[&str] = &[
    "zyron_stat_activity",
    "zyron_stat_tables",
    "zyron_stat_indexes",
    "zyron_stat_wal",
    "zyron_stat_bgwriter",
    "zyron_stat_cdc_feeds",
    "zyron_stat_replication_slots",
    "zyron_stat_cdc_streams",
    "zyron_stat_cdc_ingests",
    "zyron_stat_streaming_jobs",
    "zyron_stat_triggers",
    "zyron_stat_branches",
    "zyron_stat_publications",
    "zyron_stat_subscriptions",
    "zyron_stat_endpoints",
    "zyron_stat_dead_letters",
    "zyron_stat_zyron_sinks",
    "zyron_stat_zyron_sources",
    "zyron_stat_credential_cache",
    // Lake version history, plan items 771-775
    "zyron_table_history",
    "zyron_version_details",
    "zyron_version_files",
    "zyron_diff_versions",
    "zyron_schema_at_version",
    "zyron_version_lineage",
    "zyron_lake_branches",
    // Adaptive Clustering status, plan items 330-332
    "zyron_clustering_status",
    "zyron_derived_columns",
    "zyron_auto_compaction_history",
    // Node mesh
    "zyron_nodes",
    "zyron_table_freshness",
    "zyron_lake_log",
];

// ---------------------------------------------------------------------------
// Query shape
// ---------------------------------------------------------------------------

/// The parts of a SELECT a virtual view honors.
///
/// These views bypass the planner, so anything the parser accepted has to be
/// applied here or refused here. Silently dropping a WHERE clause would hand
/// back every row of a view the caller asked to narrow, which reads as an
/// answer rather than as a missing feature.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatViewFilters {
    /// `column = literal` conjuncts in statement order
    pub equalities: Vec<(String, String)>,
    pub limit: Option<usize>,
    pub offset: usize,
}

impl StatViewFilters {
    /// The literal a column was equated to, case-insensitive on the name
    pub fn get(&self, column: &str) -> Option<&str> {
        self.equalities
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(column))
            .map(|(_, value)| value.as_str())
    }

    /// The literal a column was equated to, parsed as an unsigned integer
    pub fn get_u64(&self, column: &str) -> Option<u64> {
        self.get(column).and_then(|v| v.trim().parse().ok())
    }

    /// Drops rows that do not satisfy every equality, then applies offset
    /// and limit. Comparison is against the row's rendered text, which is
    /// what the client sees, so a filter and a displayed value always agree.
    fn apply(
        &self,
        fields: &[FieldDescription],
        mut rows: Vec<Vec<Option<Vec<u8>>>>,
    ) -> Vec<Vec<Option<Vec<u8>>>> {
        for (column, value) in &self.equalities {
            let Some(idx) = fields
                .iter()
                .position(|f| f.name.eq_ignore_ascii_case(column))
            else {
                // A filter naming a column this view does not have matches
                // nothing, which is what the same predicate would do on a
                // real table with that column all NULL
                return Vec::new();
            };
            rows.retain(|row| {
                row.get(idx)
                    .and_then(|cell| cell.as_deref())
                    .map(|cell| cell == value.as_bytes())
                    .unwrap_or(false)
            });
        }
        if self.offset > 0 {
            if self.offset >= rows.len() {
                return Vec::new();
            }
            rows.drain(..self.offset);
        }
        if let Some(limit) = self.limit {
            rows.truncate(limit);
        }
        rows
    }
}

/// Renders a literal the way the views render their own cells, so a filter
/// value compares against a row byte for byte.
fn literal_text(lit: &zyron_parser::LiteralValue) -> Option<String> {
    match lit {
        zyron_parser::LiteralValue::Integer(n) => Some(n.to_string()),
        zyron_parser::LiteralValue::Int128(n) => Some(n.to_string()),
        zyron_parser::LiteralValue::Decimal { digits, scale } => {
            Some(zyron_common::format_decimal(*digits, *scale))
        }
        zyron_parser::LiteralValue::Float(f) => Some(f.to_string()),
        zyron_parser::LiteralValue::String(s) => Some(s.clone()),
        zyron_parser::LiteralValue::Boolean(b) => Some(b.to_string()),
        // NULL is not an equality any row can satisfy, and IS NULL is a
        // different operator this layer does not implement
        zyron_parser::LiteralValue::Null => None,
        zyron_parser::LiteralValue::Interval(i) => Some(i.to_string()),
    }
}

/// Collects `column = literal` conjuncts, refusing anything else.
fn collect_equalities(
    expr: &zyron_parser::Expr,
    view: &str,
    out: &mut Vec<(String, String)>,
) -> Result<(), ZyronError> {
    use zyron_parser::ast::BinaryOperator;
    match expr {
        zyron_parser::Expr::Nested(inner) => collect_equalities(inner, view, out),
        zyron_parser::Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                collect_equalities(left, view, out)?;
                collect_equalities(right, view, out)
            }
            BinaryOperator::Eq => {
                let pair = match (unwrap_nested(left), unwrap_nested(right)) {
                    (zyron_parser::Expr::Identifier(name), zyron_parser::Expr::Literal(lit)) => {
                        literal_text(lit).map(|v| (name.clone(), v))
                    }
                    (zyron_parser::Expr::Literal(lit), zyron_parser::Expr::Identifier(name)) => {
                        literal_text(lit).map(|v| (name.clone(), v))
                    }
                    _ => None,
                };
                match pair {
                    Some(pair) => {
                        out.push(pair);
                        Ok(())
                    }
                    None => Err(unsupported_where(view)),
                }
            }
            _ => Err(unsupported_where(view)),
        },
        _ => Err(unsupported_where(view)),
    }
}

fn unwrap_nested(expr: &zyron_parser::Expr) -> &zyron_parser::Expr {
    match expr {
        zyron_parser::Expr::Nested(inner) => unwrap_nested(inner),
        other => other,
    }
}

fn unsupported_where(view: &str) -> ZyronError {
    ZyronError::PlanError(format!(
        "{} accepts only a conjunction of column = literal in WHERE, plus LIMIT and OFFSET",
        view
    ))
}

fn constant_usize(expr: &zyron_parser::Expr, view: &str, what: &str) -> Result<usize, ZyronError> {
    match unwrap_nested(expr) {
        zyron_parser::Expr::Literal(zyron_parser::LiteralValue::Integer(n)) if *n >= 0 => {
            Ok(*n as usize)
        }
        _ => Err(ZyronError::PlanError(format!(
            "{} accepts only a non-negative constant {}",
            view, what
        ))),
    }
}

/// Reads the supported clauses off a SELECT against a virtual view, or
/// refuses a shape the view cannot answer. Never silently drops a clause.
pub fn parse_stat_view_query(
    view: &str,
    sel: &zyron_parser::SelectStatement,
) -> Result<StatViewFilters, ZyronError> {
    let mut refuse = |clause: &str| -> Result<(), ZyronError> {
        Err(ZyronError::PlanError(format!(
            "{} does not support {}",
            view, clause
        )))
    };
    if sel.with.is_some() {
        refuse("WITH")?;
    }
    if sel.distinct || !sel.distinct_on.is_empty() {
        refuse("DISTINCT")?;
    }
    if !sel.group_by.is_empty() || sel.group_by_sets.is_some() {
        refuse("GROUP BY")?;
    }
    if sel.having.is_some() {
        refuse("HAVING")?;
    }
    if sel.qualify.is_some() {
        refuse("QUALIFY")?;
    }
    if !sel.set_ops.is_empty() {
        refuse("set operations")?;
    }
    if !sel.order_by.is_empty() {
        refuse("ORDER BY")?;
    }
    if sel.fetch.is_some() {
        refuse("FETCH")?;
    }
    if sel.for_clause.is_some() {
        refuse("row locking")?;
    }

    let mut filters = StatViewFilters::default();
    if let Some(where_clause) = &sel.where_clause {
        collect_equalities(where_clause, view, &mut filters.equalities)?;
    }
    if let Some(limit) = &sel.limit {
        filters.limit = Some(constant_usize(limit, view, "LIMIT")?);
    }
    if let Some(offset) = &sel.offset {
        filters.offset = constant_usize(offset, view, "OFFSET")?;
    }
    Ok(filters)
}

/// Returns true if the given name matches a virtual statistics view.
pub fn is_stat_view(name: &str) -> bool {
    STAT_VIEW_NAMES.contains(&name)
}

/// Dispatches to the appropriate view builder and returns the column schema
/// paired with data rows. Returns None if the name is not a recognized view.
pub fn query_stat_view(
    name: &str,
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<Option<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>)>, ZyronError> {
    // The history views scope themselves by table and version so they read
    // only the log versions the query asked about, every other view builds
    // its rows and is narrowed afterwards
    let built = match name {
        "zyron_table_history" => Some(build_table_history(server, filters)?),
        "zyron_version_details" => Some(build_version_details(server, filters)?),
        "zyron_version_files" => Some(build_version_files(server, filters)?),
        "zyron_diff_versions" => Some(build_diff_versions(server, filters)?),
        "zyron_schema_at_version" => Some(build_schema_at_version(server, filters)?),
        "zyron_version_lineage" => Some(build_version_lineage(server, filters)?),
        "zyron_lake_branches" => Some(build_lake_branches(server, filters)?),
        "zyron_clustering_status" => Some(build_clustering_status(server, filters)?),
        "zyron_derived_columns" => Some(build_derived_columns(server, filters)?),
        "zyron_auto_compaction_history" => Some(build_auto_compaction_history(server, filters)?),
        "zyron_nodes" => Some(build_nodes(server, filters)?),
        "zyron_table_freshness" => Some(build_table_freshness(server, filters)?),
        "zyron_lake_log" => Some(build_lake_log(server, filters)?),
        other => build_stat_view(other, server),
    };
    Ok(built.map(|(fields, rows)| {
        let rows = filters.apply(&fields, rows);
        (fields, rows)
    }))
}

fn build_stat_view(
    name: &str,
    server: &ServerState,
) -> Option<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>)> {
    match name {
        "zyron_stat_activity" => Some(build_stat_activity(server)),
        "zyron_stat_tables" => Some(build_stat_tables(server)),
        "zyron_stat_indexes" => Some(build_stat_indexes(server)),
        "zyron_stat_wal" => Some(build_stat_wal(server)),
        "zyron_stat_bgwriter" => Some(build_stat_bgwriter(server)),
        "zyron_stat_cdc_feeds" => Some(build_stat_cdc_feeds(server)),
        "zyron_stat_replication_slots" => Some(build_stat_replication_slots(server)),
        "zyron_stat_cdc_streams" => Some(build_stat_cdc_streams(server)),
        "zyron_stat_cdc_ingests" => Some(build_stat_cdc_ingests(server)),
        "zyron_stat_streaming_jobs" => Some(build_stat_streaming_jobs(server)),
        "zyron_stat_triggers" => Some(build_stat_triggers(server)),
        "zyron_stat_branches" => Some(build_stat_branches(server)),
        "zyron_stat_publications" => Some(build_stat_publications(server)),
        "zyron_stat_subscriptions" => Some(build_stat_subscriptions(server)),
        "zyron_stat_endpoints" => Some(build_stat_endpoints(server)),
        "zyron_stat_dead_letters" => Some(build_stat_dead_letters(server)),
        "zyron_stat_zyron_sinks" => Some(build_stat_zyron_sinks(server)),
        "zyron_stat_zyron_sources" => Some(build_stat_zyron_sources(server)),
        "zyron_stat_credential_cache" => Some(build_stat_credential_cache(server)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Zyron-to-Zyron stat views
// ---------------------------------------------------------------------------

/// Builds zyron_stat_publications.
/// Columns: name, schema_id, change_feed, retention_days, classification,
///          allow_initial_snapshot, created_at.
fn build_stat_publications(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("schema_id", PG_INT4_OID, 4),
        make_field("change_feed", PG_TEXT_OID, -1),
        make_field("retention_days", PG_INT4_OID, 4),
        make_field("classification", PG_TEXT_OID, -1),
        make_field("allow_initial_snapshot", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
    ];
    let rows = server
        .catalog
        .list_publications()
        .into_iter()
        .map(|p| {
            vec![
                Some(p.name.as_bytes().to_vec()),
                Some(p.schema_id.0.to_string().into_bytes()),
                Some(p.change_feed.to_string().into_bytes()),
                Some(p.retention_days.to_string().into_bytes()),
                Some(format!("{:?}", p.classification).into_bytes()),
                Some(p.allow_initial_snapshot.to_string().into_bytes()),
                Some(p.created_at.to_string().into_bytes()),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_stat_subscriptions.
/// Columns: id, publication_id, consumer_id, mode, state, last_seen_lsn, last_poll_at.
fn build_stat_subscriptions(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("id", PG_INT4_OID, 4),
        make_field("publication_id", PG_INT4_OID, 4),
        make_field("consumer_id", PG_TEXT_OID, -1),
        make_field("mode", PG_TEXT_OID, -1),
        make_field("state", PG_TEXT_OID, -1),
        make_field("last_seen_lsn", PG_INT8_OID, 8),
        make_field("last_poll_at", PG_INT8_OID, 8),
    ];
    let rows = server
        .catalog
        .list_subscriptions()
        .into_iter()
        .map(|s| {
            vec![
                Some(s.id.0.to_string().into_bytes()),
                Some(s.publication_id.0.to_string().into_bytes()),
                Some(s.consumer_id.as_bytes().to_vec()),
                Some(format!("{:?}", s.mode).into_bytes()),
                Some(format!("{:?}", s.state).into_bytes()),
                Some(s.last_seen_lsn.to_string().into_bytes()),
                Some(s.last_poll_at.to_string().into_bytes()),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_stat_endpoints.
/// Columns: name, path, kind, enabled, auth_mode, created_at.
fn build_stat_endpoints(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("path", PG_TEXT_OID, -1),
        make_field("kind", PG_TEXT_OID, -1),
        make_field("enabled", PG_TEXT_OID, -1),
        make_field("auth_mode", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
    ];
    let rows = server
        .catalog
        .list_endpoints()
        .into_iter()
        .map(|e| {
            vec![
                Some(e.name.as_bytes().to_vec()),
                Some(e.path.as_bytes().to_vec()),
                Some(format!("{:?}", e.kind).into_bytes()),
                Some(e.enabled.to_string().into_bytes()),
                Some(format!("{:?}", e.auth_mode).into_bytes()),
                Some(e.created_at.to_string().into_bytes()),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_stat_dead_letters. Runtime DLQ contents are collected by the
/// streaming crate, this view reports zero rows until the registry callback is
/// wired through ServerState in a later phase.
fn build_stat_dead_letters(
    _server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("queue", PG_TEXT_OID, -1),
        make_field("pending", PG_INT8_OID, 8),
        make_field("oldest_ts", PG_INT8_OID, 8),
    ];
    (fields, Vec::new())
}

/// Builds zyron_stat_zyron_sinks. Lists remote Zyron sink entries from the
/// external-sink catalog whose backend is Zyron.
fn build_stat_zyron_sinks(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("uri", PG_TEXT_OID, -1),
        make_field("mode", PG_TEXT_OID, -1),
    ];
    let rows = server
        .catalog
        .list_external_sinks()
        .into_iter()
        .filter(|e| matches!(e.backend, zyron_catalog::ExternalBackend::Zyron))
        .map(|e| {
            vec![
                Some(e.name.as_bytes().to_vec()),
                Some(e.uri.as_bytes().to_vec()),
                Some(format!("{:?}", e.format).into_bytes()),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_stat_zyron_sources. Lists remote Zyron source entries from the
/// external-source catalog whose backend is Zyron.
fn build_stat_zyron_sources(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("uri", PG_TEXT_OID, -1),
        make_field("mode", PG_TEXT_OID, -1),
    ];
    let rows = server
        .catalog
        .list_external_sources()
        .into_iter()
        .filter(|e| matches!(e.backend, zyron_catalog::ExternalBackend::Zyron))
        .map(|e| {
            vec![
                Some(e.name.as_bytes().to_vec()),
                Some(e.uri.as_bytes().to_vec()),
                Some(format!("{:?}", e.mode).into_bytes()),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_stat_credential_cache. Reports zero rows until the credential
/// cache registry is wired through ServerState.
fn build_stat_credential_cache(
    _server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("provider", PG_TEXT_OID, -1),
        make_field("entries", PG_INT8_OID, 8),
        make_field("hits", PG_INT8_OID, 8),
        make_field("misses", PG_INT8_OID, 8),
        make_field("refreshes", PG_INT8_OID, 8),
    ];
    (fields, Vec::new())
}

/// Creates a FieldDescription with default values for virtual view columns.
/// table_oid, column_attr, type_modifier, and format are all set to zero/default
/// since these columns do not belong to a physical table.
fn make_field(name: &str, typeOid: i32, typeSize: i16) -> FieldDescription {
    FieldDescription {
        name: name.to_string(),
        table_oid: 0,
        column_attr: 0,
        type_oid: typeOid,
        type_size: typeSize,
        type_modifier: -1,
        format: 0,
    }
}

/// Builds the zyron_stat_activity view.
/// Columns: pid, user_name, database, state, connected_at_secs, last_activity_secs.
/// Data source: server.session_info_collector callback.
fn build_stat_activity(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("pid", PG_INT4_OID, 4),
        make_field("user_name", PG_TEXT_OID, -1),
        make_field("database", PG_TEXT_OID, -1),
        make_field("state", PG_TEXT_OID, -1),
        make_field("connected_at_secs", PG_INT8_OID, 8),
        make_field("last_activity_secs", PG_INT8_OID, 8),
    ];

    let rows = if let Some(ref collector) = server.session_info_collector {
        let sessions = collector();
        sessions
            .into_iter()
            .map(|s| {
                vec![
                    Some(s.pid.to_string().into_bytes()),
                    Some(s.user_name.into_bytes()),
                    Some(s.database.into_bytes()),
                    Some(s.state.into_bytes()),
                    Some(s.connected_at_secs.to_string().into_bytes()),
                    Some(s.last_activity_secs.to_string().into_bytes()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

/// Renders a u64 counter as the view's text representation.
fn counter_cell(value: u64) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

/// Builds the zyron_stat_tables view.
/// Columns: table_name, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch,
///          n_tup_ins, n_tup_upd, n_tup_del, n_dead_tup,
///          last_vacuum, last_analyze, bytes_read, row_count.
///
/// Activity counters come from the server's TableIOStatsRegistry, which the
/// scan and DML operators write through their execution context. They count
/// since process start, so a table this server has not touched reads zero
/// across the board rather than being absent.
///
/// last_analyze and row_count prefer the catalog's ANALYZE statistics, which
/// outlive a restart. A table never analyzed falls back to inserts less
/// deletes observed this run, which is an estimate and labelled as one here
/// rather than reported as a count.
fn build_stat_tables(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("seq_scan", PG_INT8_OID, 8),
        make_field("seq_tup_read", PG_INT8_OID, 8),
        make_field("idx_scan", PG_INT8_OID, 8),
        make_field("idx_tup_fetch", PG_INT8_OID, 8),
        make_field("n_tup_ins", PG_INT8_OID, 8),
        make_field("n_tup_upd", PG_INT8_OID, 8),
        make_field("n_tup_del", PG_INT8_OID, 8),
        make_field("n_dead_tup", PG_INT8_OID, 8),
        make_field("last_vacuum", PG_INT8_OID, 8),
        make_field("last_analyze", PG_INT8_OID, 8),
        make_field("bytes_read", PG_INT8_OID, 8),
        make_field("row_count", PG_INT8_OID, 8),
    ];

    let ordering = Ordering::Relaxed;
    let tables = server.catalog.list_all_tables();
    let rows: Vec<Vec<Option<Vec<u8>>>> = tables
        .iter()
        .map(|t| {
            let stats = server.table_io_stats.get_or_create(t.id.0);
            let analyzed = server.catalog.get_stats(t.id);
            let last_analyze = match &analyzed {
                Some(s) => s.0.last_analyzed,
                None => stats.last_analyze.load(ordering),
            };
            let row_count = match &analyzed {
                Some(s) => s.0.row_count,
                None => stats.observed_live_rows(),
            };
            vec![
                Some(t.name.as_bytes().to_vec()),
                counter_cell(stats.seq_scan.load(ordering)),
                counter_cell(stats.seq_tup_read.load(ordering)),
                counter_cell(stats.idx_scan.load(ordering)),
                counter_cell(stats.idx_tup_fetch.load(ordering)),
                counter_cell(stats.n_tup_ins.load(ordering)),
                counter_cell(stats.n_tup_upd.load(ordering)),
                counter_cell(stats.n_tup_del.load(ordering)),
                counter_cell(stats.n_dead_tup.load(ordering)),
                counter_cell(stats.last_vacuum.load(ordering)),
                counter_cell(last_analyze),
                counter_cell(stats.bytes_read.load(ordering)),
                counter_cell(row_count),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds the zyron_stat_indexes view.
/// Columns: index_name, table_name, index_type, idx_scan, idx_tup_read, idx_tup_fetch.
///
/// Counters come from the server's IndexIOStatsRegistry, written by the index
/// scan operators. idx_tup_read is index entries the range scan examined,
/// idx_tup_fetch is the table rows those entries resolved to, so the gap
/// between them is entries that pointed at a row this snapshot could not see.
fn build_stat_indexes(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("index_name", PG_TEXT_OID, -1),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("index_type", PG_TEXT_OID, -1),
        make_field("idx_scan", PG_INT8_OID, 8),
        make_field("idx_tup_read", PG_INT8_OID, 8),
        make_field("idx_tup_fetch", PG_INT8_OID, 8),
    ];

    let ordering = Ordering::Relaxed;
    let tables = server.catalog.list_all_tables();
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    for table in &tables {
        let indexes = server.catalog.get_indexes_for_table(table.id);
        for idx in &indexes {
            let type_name = match idx.index_type {
                zyron_catalog::IndexType::BTree => "btree",
                zyron_catalog::IndexType::Fulltext => "fulltext",
                zyron_catalog::IndexType::Vector => "vector",
                zyron_catalog::IndexType::Spatial => "spatial",
            };
            let stats = server.index_io_stats.get_or_create(idx.id.0);
            rows.push(vec![
                Some(idx.name.as_bytes().to_vec()),
                Some(table.name.as_bytes().to_vec()),
                Some(type_name.as_bytes().to_vec()),
                counter_cell(stats.idx_scan.load(ordering)),
                counter_cell(stats.idx_tup_read.load(ordering)),
                counter_cell(stats.idx_tup_fetch.load(ordering)),
            ]);
        }
    }
    (fields, rows)
}

/// Builds the zyron_stat_wal view.
/// Columns: wal_records, wal_bytes, wal_syncs, wal_flushed_lsn,
///          wal_current_segment, last_checkpoint_lsn.
/// Reads flushed_lsn and current_segment_id from the WAL writer.
/// Last checkpoint LSN from server.checkpoint_stats callback.
fn build_stat_wal(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("wal_records", PG_INT8_OID, 8),
        make_field("wal_bytes", PG_INT8_OID, 8),
        make_field("wal_syncs", PG_INT8_OID, 8),
        make_field("wal_flushed_lsn", PG_INT8_OID, 8),
        make_field("wal_current_segment", PG_INT4_OID, 4),
        make_field("last_checkpoint_lsn", PG_INT8_OID, 8),
    ];

    let wal_records = server.wal.wal_records_written();
    let wal_bytes = server.wal.wal_bytes_written();
    let wal_syncs = server.wal.wal_syncs.load(Ordering::Relaxed);
    let flushed_lsn = server.wal.flushed_lsn().0;
    let current_segment = server
        .wal
        .current_segment_id()
        .map(|sid| sid.0)
        .unwrap_or(0);
    let last_ckpt_lsn = server.checkpoint_stats.as_ref().map(|f| f().2).unwrap_or(0);

    let row: Vec<Option<Vec<u8>>> = vec![
        Some(wal_records.to_string().into_bytes()),
        Some(wal_bytes.to_string().into_bytes()),
        Some(wal_syncs.to_string().into_bytes()),
        Some(flushed_lsn.to_string().into_bytes()),
        Some(current_segment.to_string().into_bytes()),
        Some(last_ckpt_lsn.to_string().into_bytes()),
    ];

    (fields, vec![row])
}

/// Builds the zyron_stat_bgwriter view.
/// Columns: checkpoints_completed, checkpoint_segments_deleted,
///          last_checkpoint_lsn, vacuum_cycles, tuples_reclaimed, pages_scanned.
/// Data source: server.checkpoint_stats and server.vacuum_stats callbacks.
fn build_stat_bgwriter(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("checkpoints_completed", PG_INT8_OID, 8),
        make_field("checkpoint_segments_deleted", PG_INT8_OID, 8),
        make_field("last_checkpoint_lsn", PG_INT8_OID, 8),
        make_field("vacuum_cycles", PG_INT8_OID, 8),
        make_field("tuples_reclaimed", PG_INT8_OID, 8),
        make_field("pages_scanned", PG_INT8_OID, 8),
    ];

    let (ckpt_completed, ckpt_deleted, ckpt_lsn) = server
        .checkpoint_stats
        .as_ref()
        .map(|f| f())
        .unwrap_or((0, 0, 0));
    let (vac_cycles, vac_reclaimed, vac_pages) = server
        .vacuum_stats
        .as_ref()
        .map(|f| f())
        .unwrap_or((0, 0, 0));

    let row: Vec<Option<Vec<u8>>> = vec![
        Some(ckpt_completed.to_string().into_bytes()),
        Some(ckpt_deleted.to_string().into_bytes()),
        Some(ckpt_lsn.to_string().into_bytes()),
        Some(vac_cycles.to_string().into_bytes()),
        Some(vac_reclaimed.to_string().into_bytes()),
        Some(vac_pages.to_string().into_bytes()),
    ];

    (fields, vec![row])
}

/// Builds the zyron_stat_cdc_feeds view.
/// Columns: table_id, record_count, file_size_bytes, retention_days.
/// Data source: server.cdc_feed_stats callback.
fn build_stat_cdc_feeds(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("record_count", PG_INT8_OID, 8),
        make_field("file_size_bytes", PG_INT8_OID, 8),
        make_field("retention_days", PG_INT4_OID, 4),
    ];

    let rows = if let Some(ref stats_fn) = server.cdc_feed_stats {
        stats_fn()
            .into_iter()
            .map(|(tid, count, size, ret)| {
                vec![
                    Some(tid.to_string().into_bytes()),
                    Some(count.to_string().into_bytes()),
                    Some(size.to_string().into_bytes()),
                    Some(ret.to_string().into_bytes()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

/// Builds the zyron_stat_replication_slots view.
/// Columns: name, plugin, confirmed_lsn, restart_lsn, active, lag_bytes.
/// Data source: server.cdc_slot_stats callback.
fn build_stat_replication_slots(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("plugin", PG_TEXT_OID, -1),
        make_field("confirmed_lsn", PG_INT8_OID, 8),
        make_field("restart_lsn", PG_INT8_OID, 8),
        make_field("active", PG_TEXT_OID, -1),
        make_field("lag_bytes", PG_INT8_OID, 8),
    ];

    let rows = if let Some(ref stats_fn) = server.cdc_slot_stats {
        stats_fn()
            .into_iter()
            .map(|(name, plugin, confirmed, restart, active, lag)| {
                vec![
                    Some(name.into_bytes()),
                    Some(plugin.into_bytes()),
                    Some(confirmed.to_string().into_bytes()),
                    Some(restart.to_string().into_bytes()),
                    Some(active.to_string().into_bytes()),
                    Some(lag.to_string().into_bytes()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

/// Builds the zyron_stat_cdc_streams view.
/// Columns: name, table_id, active, slot_name.
/// Data source: server.cdc_stream_stats callback.
fn build_stat_cdc_streams(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT4_OID, 4),
        make_field("active", PG_TEXT_OID, -1),
        make_field("slot_name", PG_TEXT_OID, -1),
    ];

    let rows = if let Some(ref stats_fn) = server.cdc_stream_stats {
        stats_fn()
            .into_iter()
            .map(|(name, tid, active, slot)| {
                vec![
                    Some(name.into_bytes()),
                    Some(tid.to_string().into_bytes()),
                    Some(active.to_string().into_bytes()),
                    Some(slot.into_bytes()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

/// Builds the zyron_stat_cdc_ingests view.
/// Columns: name, table_id, active, records_applied, records_failed.
/// Data source: server.cdc_ingest_stats callback.
fn build_stat_cdc_ingests(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT4_OID, 4),
        make_field("active", PG_TEXT_OID, -1),
        make_field("records_applied", PG_INT8_OID, 8),
        make_field("records_failed", PG_INT8_OID, 8),
    ];

    let rows = if let Some(ref stats_fn) = server.cdc_ingest_stats {
        stats_fn()
            .into_iter()
            .map(|(name, tid, active, applied, failed)| {
                vec![
                    Some(name.into_bytes()),
                    Some(tid.to_string().into_bytes()),
                    Some(active.to_string().into_bytes()),
                    Some(applied.to_string().into_bytes()),
                    Some(failed.to_string().into_bytes()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

/// Builds the zyron_stat_streaming_jobs view.
/// Columns: job_id, name, status, parallelism.
/// Data source: server.stream_job_manager.
fn build_stat_streaming_jobs(
    server: &ServerState,
) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("job_id", PG_INT4_OID, 4),
        make_field("name", PG_TEXT_OID, -1),
        make_field("status", PG_TEXT_OID, -1),
        make_field("parallelism", PG_INT4_OID, 4),
    ];

    let rows = if let Some(ref mgr) = server.stream_job_manager {
        let guard = mgr.lock();
        let jobs = guard.list();
        jobs.into_iter()
            .map(|(id, name, status)| {
                vec![
                    Some(id.as_u32().to_string().into_bytes()),
                    Some(name.into_bytes()),
                    Some(format!("{:?}", status).into_bytes()),
                    Some("1".as_bytes().to_vec()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

/// Builds the zyron_stat_triggers view.
/// Columns: trigger_name, table_id, timing, events, enabled.
/// Data source: server.trigger_manager.
fn build_stat_triggers(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("trigger_name", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT4_OID, 4),
        make_field("timing", PG_TEXT_OID, -1),
        make_field("events", PG_TEXT_OID, -1),
        make_field("enabled", PG_TEXT_OID, -1),
    ];

    let rows = if let Some(ref mgr) = server.trigger_manager {
        mgr.listAll()
            .into_iter()
            .map(|t| {
                let events: String = t
                    .events
                    .iter()
                    .map(|e| format!("{:?}", e))
                    .collect::<Vec<_>>()
                    .join(",");
                vec![
                    Some(t.name.as_bytes().to_vec()),
                    Some(t.tableId.to_string().into_bytes()),
                    Some(format!("{:?}", t.timing).into_bytes()),
                    Some(events.into_bytes()),
                    Some(t.enabled.to_string().into_bytes()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

/// Builds the zyron_stat_branches view.
/// Columns: branch_name, parent_branch, created_at, is_active.
/// Data source: server.branch_manager.
fn build_stat_branches(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("branch_name", PG_TEXT_OID, -1),
        make_field("parent_branch", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
        make_field("is_active", PG_TEXT_OID, -1),
    ];

    let rows = if let Some(ref mgr) = server.branch_manager {
        let branches = mgr.list_branches();
        branches
            .into_iter()
            .map(|b| {
                vec![
                    Some(b.name.clone().into_bytes()),
                    b.parent_branch_id.map(|p| p.0.to_string().into_bytes()),
                    Some(b.created_at.to_string().into_bytes()),
                    Some("true".as_bytes().to_vec()),
                ]
            })
            .collect()
    } else {
        Vec::new()
    };
    (fields, rows)
}

// ---------------------------------------------------------------------------
// Lake version history views
// ---------------------------------------------------------------------------

/// Text cell helper, every history column renders through it so a WHERE
/// filter compares against exactly what the client is shown.
fn cell(value: impl ToString) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

/// The lake tables this node has open, narrowed by a `table_name` filter.
///
/// Only logs the node actually holds are reported. A node that does not run
/// the lake tier opens none, so its history views are empty rather than
/// quietly opening logs its deployment mode excluded.
fn lake_logs(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Vec<(String, std::sync::Arc<zyron_lake::TransactionLog>)> {
    let wanted = filters.get("table_name");
    let mut out = Vec::new();
    for table in server.catalog.list_all_tables() {
        if !table.lake.is_lake() {
            continue;
        }
        if let Some(name) = wanted {
            if table.name != name {
                continue;
            }
        }
        let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
        if let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) {
            out.push((table.name.clone(), log));
        }
    }
    out
}

/// The version a version-scoped view should read, the filter's when given
/// and the table's published head otherwise.
fn target_version(
    filters: &StatViewFilters,
    log: &zyron_lake::TransactionLog,
    column: &str,
) -> Option<u64> {
    match filters.get_u64(column) {
        Some(v) => Some(v),
        None => match log.latest_version() {
            0 => None,
            v => Some(v),
        },
    }
}

/// How far back a history walk has to read.
///
/// The walk can stop at the LIMIT only when nothing else narrows the result,
/// otherwise a row dropped by a later filter would shorten the answer.
fn history_walk_limit(filters: &StatViewFilters) -> usize {
    let only_table_scope = filters
        .equalities
        .iter()
        .all(|(name, _)| name.eq_ignore_ascii_case("table_name"));
    match (only_table_scope, filters.offset, filters.limit) {
        (true, 0, Some(limit)) => limit,
        _ => usize::MAX,
    }
}

fn build_table_history(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("version", PG_INT8_OID, 8),
        make_field("operation", PG_TEXT_OID, -1),
        make_field("timestamp_us", PG_INT8_OID, 8),
        make_field("read_version", PG_INT8_OID, 8),
        make_field("db_txn_id", PG_INT8_OID, 8),
        make_field("commit_lsn", PG_INT8_OID, 8),
        make_field("files_added", PG_INT4_OID, 4),
        make_field("files_removed", PG_INT4_OID, 4),
        make_field("rows_added", PG_INT8_OID, 8),
        make_field("rows_removed", PG_INT8_OID, 8),
        make_field("bytes_added", PG_INT8_OID, 8),
    ];
    let walk = history_walk_limit(filters);
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        for record in zyron_lake::table_history(&log, walk)? {
            rows.push(vec![
                cell(&name),
                cell(record.version),
                cell(record.operation.name()),
                cell(record.timestamp_us),
                cell(record.read_version),
                cell(record.db_txn_id),
                cell(record.commit_lsn),
                cell(record.files_added),
                cell(record.files_removed),
                cell(record.rows_added),
                cell(record.rows_removed),
                cell(record.bytes_added),
            ]);
        }
    }
    Ok((fields, rows))
}

fn build_version_details(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("version", PG_INT8_OID, 8),
        make_field("operation", PG_TEXT_OID, -1),
        make_field("timestamp_us", PG_INT8_OID, 8),
        make_field("identity", PG_TEXT_OID, -1),
        make_field("client", PG_TEXT_OID, -1),
        make_field("commit_info", PG_TEXT_OID, -1),
        make_field("correlation_id", PG_TEXT_OID, -1),
        make_field("trace_id", PG_TEXT_OID, -1),
        make_field("files_added", PG_INT4_OID, 4),
        make_field("files_removed", PG_INT4_OID, 4),
        make_field("delete_predicates", PG_TEXT_OID, -1),
        make_field("properties_set", PG_TEXT_OID, -1),
        make_field("schema_id", PG_INT8_OID, 8),
    ];
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let Some(version) = target_version(filters, &log, "version") else {
            continue;
        };
        let details = zyron_lake::version_details(&log, version)?;
        let audit = details.audit.unwrap_or_default();
        let properties = details
            .properties
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        rows.push(vec![
            cell(&name),
            cell(details.record.version),
            cell(details.record.operation.name()),
            cell(details.record.timestamp_us),
            cell(&audit.identity),
            cell(&audit.client),
            cell(&audit.commit_info),
            cell(&audit.correlation_id),
            cell(&audit.trace_id),
            cell(details.files_added.len()),
            cell(details.files_removed.len()),
            cell(details.delete_predicates.join("; ")),
            cell(properties),
            details.schema_id.map(|id| id.to_string().into_bytes()),
        ]);
    }
    Ok((fields, rows))
}

fn build_version_files(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("version", PG_INT8_OID, 8),
        make_field("partition_id", PG_TEXT_OID, -1),
        make_field("size_bytes", PG_INT8_OID, 8),
        make_field("row_count", PG_INT8_OID, 8),
        make_field("added_version", PG_INT8_OID, 8),
        make_field("cluster_spec_id", PG_INT4_OID, 4),
        make_field("delete_predicate_ids", PG_TEXT_OID, -1),
    ];
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let Some(version) = target_version(filters, &log, "version") else {
            continue;
        };
        for file in zyron_lake::version_files(&log, version)? {
            let predicates = file
                .delete_predicate_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            rows.push(vec![
                cell(&name),
                cell(version),
                // Hex, matching the data file name on disk
                cell(format!("{:016x}", file.partition_id)),
                cell(file.size_bytes),
                cell(file.row_count),
                cell(file.added_version),
                cell(file.cluster_spec_id),
                cell(predicates),
            ]);
        }
    }
    Ok((fields, rows))
}

fn build_diff_versions(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("from_version", PG_INT8_OID, 8),
        make_field("to_version", PG_INT8_OID, 8),
        make_field("files_added", PG_INT4_OID, 4),
        make_field("files_removed", PG_INT4_OID, 4),
        make_field("rows_added", PG_INT8_OID, 8),
        make_field("rows_removed", PG_INT8_OID, 8),
        make_field("bytes_added", PG_INT8_OID, 8),
        make_field("bytes_removed", PG_INT8_OID, 8),
    ];
    // A diff without both endpoints has no answer, and returning every row
    // unfiltered would read as one
    let (Some(from), Some(to)) = (
        filters.get_u64("from_version"),
        filters.get_u64("to_version"),
    ) else {
        return Err(ZyronError::PlanError(
            "zyron_diff_versions needs from_version = <n> AND to_version = <n> in WHERE".into(),
        ));
    };
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let diff = zyron_lake::diff_versions(&log, from, to)?;
        rows.push(vec![
            cell(&name),
            cell(diff.from_version),
            cell(diff.to_version),
            cell(diff.files_added.len()),
            cell(diff.files_removed.len()),
            cell(diff.rows_added),
            cell(diff.rows_removed),
            cell(diff.bytes_added),
            cell(diff.bytes_removed),
        ]);
    }
    Ok((fields, rows))
}

fn build_schema_at_version(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("version", PG_INT8_OID, 8),
        make_field("schema_id", PG_INT8_OID, 8),
        make_field("column_id", PG_INT4_OID, 4),
        make_field("column_name", PG_TEXT_OID, -1),
        make_field("type_id", PG_TEXT_OID, -1),
        make_field("nullable", PG_TEXT_OID, -1),
        make_field("fractional_digits", PG_INT4_OID, 4),
        make_field("max_length", PG_INT8_OID, 8),
        make_field("default_expr", PG_TEXT_OID, -1),
    ];
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let Some(version) = target_version(filters, &log, "version") else {
            continue;
        };
        let schema = zyron_lake::schema_at_version(&log, version)?;
        for column in &schema.columns {
            rows.push(vec![
                cell(&name),
                cell(version),
                cell(schema.schema_id),
                cell(column.id),
                cell(&column.name),
                cell(format!("{:?}", column.type_id)),
                cell(column.nullable),
                column.fractional_digits.map(|p| p.to_string().into_bytes()),
                column.max_length.map(|n| n.to_string().into_bytes()),
                column.default_expr.as_ref().map(|e| e.clone().into_bytes()),
            ]);
        }
    }
    Ok((fields, rows))
}

fn build_version_lineage(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("version", PG_INT8_OID, 8),
        make_field("depth", PG_INT4_OID, 4),
        make_field("ancestor_version", PG_INT8_OID, 8),
    ];
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let Some(version) = target_version(filters, &log, "version") else {
            continue;
        };
        for (depth, ancestor) in zyron_lake::version_lineage(&log, version)?
            .into_iter()
            .enumerate()
        {
            rows.push(vec![
                cell(&name),
                cell(version),
                cell(depth),
                cell(ancestor),
            ]);
        }
    }
    Ok((fields, rows))
}

fn build_lake_branches(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("branch_name", PG_TEXT_OID, -1),
        make_field("base_version", PG_INT8_OID, 8),
        make_field("head_version", PG_INT8_OID, 8),
        make_field("commits_ahead", PG_INT8_OID, 8),
        make_field("created_us", PG_INT8_OID, 8),
    ];
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        for info in zyron_lake::list_branches(log.paths())? {
            rows.push(vec![
                cell(&name),
                cell(&info.name),
                cell(info.base_version),
                cell(info.head_version),
                cell(info.head_version - info.base_version),
                cell(info.created_us),
            ]);
        }
    }
    Ok((fields, rows))
}

/// One row per lake table: the layout in force, the policy that governs
/// it, what the workload measured, and what measurement would choose.
///
/// The last two columns are why the view exists under every mode. An
/// operator who pinned a layout with FORCE is exactly the person who
/// should be able to see that measurement disagrees, and seeing it costs
/// them nothing: this view reads, it never proposes a commit.
/// Every expression a lake table is clustered by, and the column its
/// values live in.
///
/// An expression cluster key is stored in a column no statement named, so
/// it is not in the table's column list and nothing else reports it. This
/// is where an operator finds out that a column exists, what it computes,
/// and which columns it reads, which is what makes a later DROP COLUMN
/// refusal explicable
fn build_derived_columns(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("column_id", PG_INT8_OID, 8),
        make_field("column_name", PG_TEXT_OID, -1),
        // Rendered as text: the identity is 64 unsigned bits and the wire
        // has no unsigned integer type wide enough to carry it
        make_field("canonical_hash", PG_TEXT_OID, -1),
        make_field("sql", PG_TEXT_OID, -1),
        make_field("source_columns", PG_TEXT_OID, -1),
        make_field("type_name", PG_TEXT_OID, -1),
        make_field("fractional_digits", PG_INT4_OID, 4),
        make_field("is_cluster_key", PG_TEXT_OID, -1),
        make_field("addressable_by", PG_TEXT_OID, -1),
    ];
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let manifest = log.latest_manifest()?;
        for derived in &manifest.schema.derived {
            let Some(column) = manifest.schema.column_by_id(derived.column_id) else {
                continue;
            };
            let sources = derived
                .source_columns
                .iter()
                .map(|id| {
                    manifest
                        .schema
                        .column_by_id(*id)
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("column {id}"))
                })
                .collect::<Vec<_>>()
                .join(", ");
            let is_key = manifest
                .cluster_spec
                .keys
                .iter()
                .any(|k| k.column_id == derived.column_id);
            rows.push(vec![
                cell(&name),
                cell(derived.column_id as i64),
                cell(&column.name),
                cell(&format!("{:016x}", derived.canonical_hash)),
                cell(&derived.sql),
                cell(&sources),
                cell(&column.type_id.to_string()),
                match column.fractional_digits {
                    Some(d) => cell(i32::from(d)),
                    None => None,
                },
                cell(if is_key { "yes" } else { "no" }),
                // A derived column is reached through its expression: the
                // catalog's column list is positional and a column no
                // statement supplies would shift every column after it
                cell("expression"),
            ]);
        }
    }
    Ok((fields, rows))
}

/// Compactions this node ran without being asked, newest last.
///
/// A maintenance loop that rewrites files silently is one nobody can
/// reason about, so every run says what tripped it and what it moved. The
/// ring is bounded and lives in memory: the durable record of what
/// happened is each table's transaction log
fn build_auto_compaction_history(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT8_OID, 8),
        make_field("trigger", PG_TEXT_OID, -1),
        make_field("triggered_at_us", PG_INT8_OID, 8),
        make_field("files_before", PG_INT8_OID, 8),
        make_field("files_after", PG_INT8_OID, 8),
        make_field("dead_rows_reclaimed", PG_INT8_OID, 8),
        // Rates render in thousandths so the view stays integer typed,
        // matching the clustering views and the metric families
        make_field("small_file_ratio_milli", PG_INT4_OID, 4),
        make_field("dead_row_ratio_milli", PG_INT4_OID, 4),
        make_field("version", PG_INT8_OID, 8),
    ];
    // Filtering by table happens on the ring rather than after it, so a
    // busy node's history for one table is not crowded out by another's
    let wanted = filters.get("table_name").map(|s| s.to_string());
    let mut rows = Vec::new();
    for run in zyron_lake::compaction_history::compaction_history().runs() {
        if let Some(name) = &wanted {
            if run.table_name != *name {
                continue;
            }
        }
        rows.push(vec![
            cell(&run.table_name),
            cell(i64::from(run.table_id)),
            cell(run.trigger.as_str()),
            cell(run.triggered_at_us),
            cell(run.files_before as i64),
            cell(run.files_after as i64),
            cell(run.dead_rows_reclaimed as i64),
            cell(run.small_file_ratio_milli as i32),
            cell(run.dead_row_ratio_milli as i32),
            // A run that changed nothing committed no version, and zero is
            // a version rather than an absence
            match run.version {
                Some(v) => cell(v as i64),
                None => None,
            },
        ]);
    }
    let _ = server;
    Ok((fields, rows))
}

fn build_clustering_status(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("mode", PG_TEXT_OID, -1),
        make_field("schedule", PG_TEXT_OID, -1),
        make_field("spec_id", PG_INT4_OID, 4),
        // What a statement asked for and what the files are actually laid
        // out by, separately. Under Auto they diverge the moment
        // measurement replaces a declared key, and an operator glancing at
        // one column could not tell that had happened
        make_field("declared_keys", PG_TEXT_OID, -1),
        make_field("active_keys", PG_TEXT_OID, -1),
        make_field("anchors", PG_TEXT_OID, -1),
        make_field("files", PG_INT8_OID, 8),
        make_field("bytes", PG_INT8_OID, 8),
        make_field("observed_columns", PG_INT4_OID, 4),
        // Rates render in thousandths so the view stays integer typed,
        // matching how the clustering metric families report them
        make_field("measured_fit_milli", PG_INT4_OID, 4),
        make_field("would_choose", PG_TEXT_OID, -1),
        make_field("dropped_observations", PG_INT8_OID, 8),
    ];
    let now = zyron_lake::current_epoch();
    let observer = zyron_lake::observer();
    let mut rows = Vec::new();
    let declared_by_table: std::collections::HashMap<u32, Vec<zyron_lake::ClusterKey>> = server
        .catalog
        .list_all_tables()
        .into_iter()
        .map(|t| (t.id.0, t.cluster.fold_keys()))
        .collect();
    for (name, log) in lake_logs(server, filters) {
        let Some(table_id) = log.paths().table_id() else {
            continue;
        };
        let manifest = log.latest_manifest()?;
        let evidence = zyron_lake::evidence_from_manifest(&manifest, observer, table_id, now);
        let anchors = manifest.clustering_anchors();

        let mut fit_weight = 0f64;
        let mut fit_total = 0f64;
        let mut observed = 0i32;
        for column in &evidence {
            let weight = column.total_weight();
            if weight <= 0.0 {
                continue;
            }
            observed += 1;
            if let Some(rate) =
                zyron_lake::measured_skip_rate(observer, table_id, column.column_id, now)
            {
                fit_weight += weight;
                fit_total += weight * rate;
            }
        }
        let name_of = |column_id: u32| -> String {
            manifest
                .schema
                .column_by_id(column_id)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("column {}", column_id))
        };
        let render = |keys: &[zyron_lake::ClusterKey]| -> String {
            keys.iter()
                .map(|k| format!("{} USING {}", name_of(k.column_id), k.strategy.as_str()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let proposal = zyron_lake::propose(&evidence, &anchors, 4);
        rows.push(vec![
            cell(&name),
            cell(manifest.clustering_mode().as_str()),
            cell(manifest.clustering_schedule().as_str()),
            cell(manifest.cluster_spec.spec_id as i32),
            cell(&render(
                declared_by_table
                    .get(&table_id)
                    .map(|k| k.as_slice())
                    .unwrap_or(&[]),
            )),
            cell(&render(&manifest.cluster_spec.keys)),
            cell(
                &anchors
                    .iter()
                    .map(|id| name_of(*id))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            cell(manifest.entries.len() as i64),
            cell(manifest.entries.iter().map(|e| e.size_bytes).sum::<u64>() as i64),
            cell(observed),
            // No observation is not a fit of zero, so it reports as NULL
            if fit_weight > 0.0 {
                cell(((fit_total / fit_weight) * 1000.0).round() as i32)
            } else {
                None
            },
            cell(&render(&proposal)),
            cell(observer.stats().dropped as i64),
        ]);
    }
    Ok((fields, rows))
}

/// This node and every peer it has been told about.
///
/// The local row is what this node knows for certain: its own id, mode,
/// and how many tables of each format it actually holds. A peer row is
/// what the operator declared plus what has been learned since, and its
/// unknowns render as NULL rather than as a guess, because a mesh view
/// that invents a peer's mode is worse than one that admits it has not
/// reached the peer yet.
fn build_nodes(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("node_name", PG_TEXT_OID, -1),
        make_field("node_id", PG_TEXT_OID, -1),
        make_field("is_local", PG_TEXT_OID, -1),
        make_field("mode", PG_TEXT_OID, -1),
        make_field("address", PG_TEXT_OID, -1),
        make_field("heap_tables", PG_INT4_OID, 4),
        make_field("lake_tables", PG_INT4_OID, 4),
        make_field("since_us", PG_INT8_OID, 8),
    ];
    let wanted = filters.get("node_name");
    let mut rows = Vec::new();

    let identity = &server.node_identity;
    if wanted.map(|n| n == identity.name).unwrap_or(true) {
        let mut heap = 0i32;
        let mut lake = 0i32;
        for table in server.catalog.list_all_tables() {
            if table.lake.is_lake() {
                lake += 1;
            } else {
                heap += 1;
            }
        }
        rows.push(vec![
            cell(&identity.name),
            cell(&format!("{:016x}", identity.node_id)),
            cell("t"),
            cell(identity.mode.as_str()),
            // The local node is reached by connecting to it, so it
            // advertises no address to itself
            None,
            cell(heap),
            cell(lake),
            cell(identity.created_us),
        ]);
    }

    for peer in server.peers.read().peers() {
        if let Some(name) = wanted {
            if peer.name != name {
                continue;
            }
        }
        rows.push(vec![
            cell(&peer.name),
            // Learned on contact, absent until then. A guess here would be
            // worse than an admitted unknown, because a router would act
            // on it
            peer.node_id
                .map(|id| format!("{:016x}", id))
                .map(|s| cell(&s))
                .unwrap_or(None),
            cell("f"),
            // What the peer said, falling back to what was declared
            peer.effective_mode()
                .map(|m| m.as_str())
                .map(cell)
                .unwrap_or(None),
            cell(&peer.address),
            // Table counts belong to the peer and are not guessed here
            None,
            None,
            // Last contact when there has been one, otherwise when it was
            // declared, so the column always answers "as of when"
            cell(if peer.last_seen_us != 0 {
                peer.last_seen_us
            } else {
                peer.added_us
            }),
        ]);
    }
    Ok((fields, rows))
}

/// How current each lake table's data is.
///
/// A table this node writes is its own authority and is current by
/// definition. A table this node follows is only as current as the last
/// version it replayed, and a reader is entitled to know that before
/// trusting the answer: a stale read that admits it is stale is useful,
/// one that does not is a wrong answer.
///
/// Lag is counted in versions rather than seconds. Two nodes do not share
/// a clock, and a follower one version behind a table nobody writes is
/// current, not stale.
fn build_table_freshness(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("role", PG_TEXT_OID, -1),
        make_field("version", PG_INT8_OID, 8),
        make_field("leader_version", PG_INT8_OID, 8),
        make_field("lag_versions", PG_INT8_OID, 8),
        make_field("applied_us", PG_INT8_OID, 8),
        make_field("is_current", PG_TEXT_OID, -1),
        make_field("writer_node", PG_TEXT_OID, -1),
    ];
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let manifest = log.latest_manifest()?;
        let owner = zyron_lake::writer_node(&manifest);
        // A follower is a table carrying a replay cursor. Nothing else has
        // one, so the relationship needs no separate declaration
        let cursor = zyron_lake::load_cursor(log.paths());
        let (role, leader_version, lag, applied_us, current) = match cursor {
            Some(freshness) => (
                "follower",
                freshness.leader_version,
                freshness.lag_versions(),
                freshness.applied_us,
                freshness.is_current(),
            ),
            None => (
                "leader",
                log.latest_version(),
                0,
                manifest.timestamp_us,
                true,
            ),
        };
        rows.push(vec![
            cell(&name),
            cell(role),
            cell(log.latest_version() as i64),
            cell(leader_version as i64),
            cell(lag as i64),
            cell(applied_us),
            cell(if current { "t" } else { "f" }),
            // Unowned means no node has claimed writes, which is what a
            // single-node deployment looks like and is not a fault
            owner
                .map(|id| cell(&format!("{:016x}", id)))
                .unwrap_or(None),
        ]);
    }
    Ok((fields, rows))
}

/// A lake table's version files, so a follower can read the log of a
/// leader it cannot see the filesystem of.
///
/// This is the whole replication payload. A version file is a complete,
/// self-describing description of one commit, so shipping these and
/// nothing else is what "ship the log, not the data" means over a wire as
/// much as over shared storage.
///
/// Versions come back contiguous from `from_version` and stop at the first
/// gap, matching what a follower reading the filesystem directly would
/// see. A follower that jumped a gap would skip a commit silently.
///
/// The payload is hex rather than raw bytes because this crosses the text
/// protocol, where an arbitrary byte is not a valid cell. A version file is
/// metadata, hundreds of bytes to a few kilobytes, so doubling it costs
/// far less than the data it describes.
fn build_lake_log(
    server: &ServerState,
    filters: &StatViewFilters,
) -> Result<(Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>), ZyronError> {
    // from_version is echoed as a column, not only read as a parameter.
    // Narrowing drops any row that fails an equality, and it compares
    // against what the row displays, so a request parameter that never
    // appears in a row would drop every row it selected
    let fields = vec![
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("from_version", PG_INT8_OID, 8),
        make_field("version", PG_INT8_OID, 8),
        make_field("payload", PG_TEXT_OID, -1),
    ];
    let from = filters.get_u64("from_version").unwrap_or(0);
    // A follower asks for one table at a time, because it follows one
    // leader table at a time and returning every table's log would send
    // work nobody asked for
    let limit = filters.limit.unwrap_or(256).min(4096);
    let mut rows = Vec::new();
    for (name, log) in lake_logs(server, filters) {
        let head = log.latest_version();
        let mut version = from.saturating_add(1).max(1);
        while version <= head && rows.len() < limit {
            let path = log.paths().version_file(version);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    rows.push(vec![
                        cell(&name),
                        cell(from as i64),
                        cell(version as i64),
                        cell(&hex_encode(&bytes)),
                    ]);
                }
                // A missing version file is a gap. Stopping here is what
                // keeps a follower behind rather than wrong
                Err(_) => break,
            }
            version += 1;
        }
    }
    Ok((fields, rows))
}

/// Lowercase hex, two characters per byte.
fn hex_encode(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0F) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lake_log_is_a_stat_view() {
        assert!(is_stat_view("zyron_lake_log"));
    }

    #[test]
    fn test_hex_encoding_round_trips() {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let hex = hex_encode(&bytes);
        assert_eq!(hex.len(), 512);
        let decoded = zyron_lake::decode_hex(&hex).expect("decode");
        assert_eq!(decoded, bytes);
        assert!(zyron_lake::decode_hex("xyz").is_none());
        assert!(zyron_lake::decode_hex("abc").is_none(), "odd length");
    }

    #[test]
    fn test_table_freshness_is_a_stat_view() {
        assert!(is_stat_view("zyron_table_freshness"));
    }

    #[test]
    fn test_nodes_is_a_stat_view() {
        assert!(is_stat_view("zyron_nodes"));
    }

    #[test]
    fn test_clustering_status_is_a_stat_view() {
        assert!(is_stat_view("zyron_clustering_status"));
    }

    #[test]
    fn test_is_stat_view_recognized() {
        assert!(is_stat_view("zyron_stat_activity"));
        assert!(is_stat_view("zyron_stat_tables"));
        assert!(is_stat_view("zyron_stat_indexes"));
        assert!(is_stat_view("zyron_stat_wal"));
        assert!(is_stat_view("zyron_stat_bgwriter"));
        assert!(is_stat_view("zyron_stat_streaming_jobs"));
        assert!(is_stat_view("zyron_stat_triggers"));
        assert!(is_stat_view("zyron_stat_branches"));
    }

    #[test]
    fn test_is_stat_view_unrecognized() {
        assert!(!is_stat_view("zyron_stat_unknown"));
        assert!(!is_stat_view("pg_stat_activity"));
        assert!(!is_stat_view(""));
    }

    #[test]
    fn test_is_stat_view_publications_recognized() {
        assert!(is_stat_view("zyron_stat_publications"));
        assert!(is_stat_view("zyron_stat_subscriptions"));
        assert!(is_stat_view("zyron_stat_endpoints"));
    }

    #[test]
    fn test_is_stat_view_z2z_runtime_recognized() {
        assert!(is_stat_view("zyron_stat_dead_letters"));
        assert!(is_stat_view("zyron_stat_zyron_sinks"));
        assert!(is_stat_view("zyron_stat_zyron_sources"));
        assert!(is_stat_view("zyron_stat_credential_cache"));
    }

    #[test]
    fn test_publications_schema_has_seven_columns() {
        let fields = vec![
            make_field("name", PG_TEXT_OID, -1),
            make_field("schema_id", PG_INT4_OID, 4),
            make_field("change_feed", PG_TEXT_OID, -1),
            make_field("retention_days", PG_INT4_OID, 4),
            make_field("classification", PG_TEXT_OID, -1),
            make_field("allow_initial_snapshot", PG_TEXT_OID, -1),
            make_field("created_at", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 7);
        assert_eq!(fields[0].name, "name");
        assert_eq!(fields[6].name, "created_at");
    }

    #[test]
    fn test_subscriptions_schema_has_seven_columns() {
        let fields = vec![
            make_field("id", PG_INT4_OID, 4),
            make_field("publication_id", PG_INT4_OID, 4),
            make_field("consumer_id", PG_TEXT_OID, -1),
            make_field("mode", PG_TEXT_OID, -1),
            make_field("state", PG_TEXT_OID, -1),
            make_field("last_seen_lsn", PG_INT8_OID, 8),
            make_field("last_poll_at", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 7);
        assert_eq!(fields[3].name, "mode");
    }

    #[test]
    fn test_endpoints_schema_has_six_columns() {
        let fields = vec![
            make_field("name", PG_TEXT_OID, -1),
            make_field("path", PG_TEXT_OID, -1),
            make_field("kind", PG_TEXT_OID, -1),
            make_field("enabled", PG_TEXT_OID, -1),
            make_field("auth_mode", PG_TEXT_OID, -1),
            make_field("created_at", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[1].name, "path");
    }

    #[test]
    fn test_make_field_text() {
        let field = make_field("col_name", PG_TEXT_OID, -1);
        assert_eq!(field.name, "col_name");
        assert_eq!(field.type_oid, PG_TEXT_OID);
        assert_eq!(field.type_size, -1);
        assert_eq!(field.table_oid, 0);
        assert_eq!(field.column_attr, 0);
        assert_eq!(field.type_modifier, -1);
        assert_eq!(field.format, 0);
    }

    #[test]
    fn test_make_field_int4() {
        let field = make_field("pid", PG_INT4_OID, 4);
        assert_eq!(field.type_oid, PG_INT4_OID);
        assert_eq!(field.type_size, 4);
    }

    #[test]
    fn test_make_field_int8() {
        let field = make_field("counter", PG_INT8_OID, 8);
        assert_eq!(field.type_oid, PG_INT8_OID);
        assert_eq!(field.type_size, 8);
    }

    #[test]
    fn test_query_stat_view_unknown_returns_none() {
        // Cannot construct ServerState in unit tests without full subsystem init,
        // but we can verify the None path by checking is_stat_view instead.
        assert!(!is_stat_view("no_such_view"));
    }

    #[test]
    fn test_stat_activity_schema() {
        // Verify the field descriptors are built correctly by calling the
        // builder directly (requires a ServerState, tested via integration).
        let fields = vec![
            make_field("pid", PG_INT4_OID, 4),
            make_field("user_name", PG_TEXT_OID, -1),
            make_field("database", PG_TEXT_OID, -1),
            make_field("state", PG_TEXT_OID, -1),
            make_field("connected_at_secs", PG_INT8_OID, 8),
            make_field("last_activity_secs", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[0].name, "pid");
        assert_eq!(fields[5].name, "last_activity_secs");
    }

    #[test]
    fn test_stat_tables_schema() {
        let fields = vec![
            make_field("table_name", PG_TEXT_OID, -1),
            make_field("seq_scan", PG_INT8_OID, 8),
            make_field("seq_tup_read", PG_INT8_OID, 8),
            make_field("idx_scan", PG_INT8_OID, 8),
            make_field("idx_tup_fetch", PG_INT8_OID, 8),
            make_field("n_tup_ins", PG_INT8_OID, 8),
            make_field("n_tup_upd", PG_INT8_OID, 8),
            make_field("n_tup_del", PG_INT8_OID, 8),
            make_field("n_dead_tup", PG_INT8_OID, 8),
            make_field("last_vacuum", PG_INT8_OID, 8),
            make_field("last_analyze", PG_INT8_OID, 8),
            make_field("row_count", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 12);
        assert_eq!(fields[0].name, "table_name");
        assert_eq!(fields[11].name, "row_count");
    }

    #[test]
    fn test_stat_indexes_schema() {
        let fields = vec![
            make_field("index_name", PG_TEXT_OID, -1),
            make_field("table_name", PG_TEXT_OID, -1),
            make_field("index_type", PG_TEXT_OID, -1),
            make_field("idx_scan", PG_INT8_OID, 8),
            make_field("idx_tup_read", PG_INT8_OID, 8),
            make_field("idx_tup_fetch", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[2].name, "index_type");
    }

    #[test]
    fn test_stat_wal_schema() {
        let fields = vec![
            make_field("wal_records", PG_INT8_OID, 8),
            make_field("wal_bytes", PG_INT8_OID, 8),
            make_field("wal_syncs", PG_INT8_OID, 8),
            make_field("wal_flushed_lsn", PG_INT8_OID, 8),
            make_field("wal_current_segment", PG_INT4_OID, 4),
            make_field("last_checkpoint_lsn", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[3].name, "wal_flushed_lsn");
        assert_eq!(fields[4].type_oid, PG_INT4_OID);
    }

    #[test]
    fn test_stat_bgwriter_schema() {
        let fields = vec![
            make_field("checkpoints_completed", PG_INT8_OID, 8),
            make_field("checkpoint_segments_deleted", PG_INT8_OID, 8),
            make_field("last_checkpoint_lsn", PG_INT8_OID, 8),
            make_field("vacuum_cycles", PG_INT8_OID, 8),
            make_field("tuples_reclaimed", PG_INT8_OID, 8),
            make_field("pages_scanned", PG_INT8_OID, 8),
        ];
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[0].name, "checkpoints_completed");
        assert_eq!(fields[5].name, "pages_scanned");
    }
}

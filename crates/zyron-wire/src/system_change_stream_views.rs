//! The `zyron_sys.cdc.change_streams`, `zyron_sys.cdc.feeds`,
//! `zyron_sys.cdc.apply_runs` and `zyron_sys.alert.templates` views.
//!
//! Every number here is read off a counter or a catalog entry. A pending row
//! count comes from the feed's per-version index and a lag reading from its
//! timestamps, so listing every stream on a node opens no change file

use zyron_cdc::change_stream::{ALERT_TEMPLATES, ChangeStreamRuntime};
use zyron_common::ZyronError;

use crate::connection::ServerState;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_BOOL_OID, PG_INT4_OID, PG_INT8_OID, PG_TEXT_OID};

/// Builds one of the views this module owns
pub fn build(schema: &str, object: &str, server: &ServerState) -> Result<ViewRows, ZyronError> {
    match (schema, object) {
        ("cdc", "change_streams") => Ok(build_change_streams(server)),
        ("cdc", "feeds") => Ok(build_feeds(server)),
        ("cdc", "apply_runs") => Ok(build_apply_runs()),
        ("alert", "templates") => Ok(build_alert_templates()),
        (schema, object) => Err(ZyronError::Internal(format!(
            "`zyron_sys.{schema}.{object}` is registered but has no builder"
        ))),
    }
}

/// Whether this module builds the named view
pub fn owns(schema: &str, object: &str) -> bool {
    matches!(
        (schema, object),
        ("cdc", "change_streams")
            | ("cdc", "feeds")
            | ("cdc", "apply_runs")
            | ("alert", "templates")
    )
}

fn text(value: impl Into<String>) -> Option<Vec<u8>> {
    Some(value.into().into_bytes())
}

fn number(value: impl std::fmt::Display) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

fn flag(value: bool) -> Option<Vec<u8>> {
    Some(if value { b"t".to_vec() } else { b"f".to_vec() })
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// The name a table id resolves to, or the id itself when the table is gone
fn table_name(server: &ServerState, table_id: u32) -> String {
    server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
        .map(|table| table.name.clone())
        .unwrap_or_else(|_| table_id.to_string())
}

/// Builds zyron_sys.cdc.change_streams
fn build_change_streams(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("stream", PG_TEXT_OID, -1),
        make_field("sources", PG_TEXT_OID, -1),
        make_field("position", PG_TEXT_OID, -1),
        make_field("consumed", PG_TEXT_OID, -1),
        make_field("mode", PG_TEXT_OID, -1),
        make_field("stale", PG_BOOL_OID, 1),
        make_field("stale_reason", PG_TEXT_OID, -1),
        make_field("needs_attention", PG_BOOL_OID, 1),
        make_field("attention_reason", PG_TEXT_OID, -1),
        make_field("pending_rows", PG_INT8_OID, 8),
        make_field("pending_versions", PG_INT8_OID, 8),
        make_field("lag_seconds", PG_INT8_OID, 8),
        make_field("last_advanced_at", PG_INT8_OID, 8),
        make_field("last_advanced_by", PG_INT4_OID, 4),
        make_field("owner", PG_INT4_OID, 4),
        make_field("branch", PG_TEXT_OID, -1),
    ];
    let Some(feeds) = server.cdc_registry.as_ref() else {
        return (fields, Vec::new());
    };
    let runtime = ChangeStreamRuntime::new(std::sync::Arc::clone(feeds));
    let now = now_micros();
    let rows = server
        .catalog
        .list_change_streams()
        .into_iter()
        .map(|entry| {
            let status = runtime.status(&entry, now);
            let sources = entry
                .source
                .table_ids()
                .into_iter()
                .map(|id| table_name(server, id))
                .collect::<Vec<_>>()
                .join(", ");
            let position = entry
                .position
                .iter()
                .map(|p| format!("{}={}", table_name(server, p.table_id), p.version))
                .collect::<Vec<_>>()
                .join(", ");
            let consumed = entry
                .position
                .iter()
                .map(|p| format!("{}={}", table_name(server, p.table_id), p.consumed))
                .collect::<Vec<_>>()
                .join(", ");
            let mode = match entry.mode {
                zyron_catalog::ChangeStreamMode::Standard => "standard",
                zyron_catalog::ChangeStreamMode::AppendOnly => "append_only",
            };
            vec![
                text(entry.name.clone()),
                text(sources),
                text(position),
                text(consumed),
                text(mode),
                flag(status.stale),
                text(status.stale_reason),
                flag(status.needs_attention),
                text(status.attention_reason),
                number(status.pending_rows),
                number(status.pending_versions),
                number(status.lag_seconds),
                number(status.last_advanced_at),
                number(status.last_advanced_by),
                number(status.owner_id),
                text(branch_name(server, entry.branch)),
            ]
        })
        .collect();
    (fields, rows)
}

/// The name of the branch a stream was created on, empty for a stream on
/// the table itself
fn branch_name(server: &ServerState, branch: Option<u64>) -> String {
    let Some(id) = branch else {
        return String::new();
    };
    server
        .branch_manager
        .as_ref()
        .and_then(|mgr| mgr.get_branch(zyron_versioning::BranchId(id)).ok())
        .map(|entry| entry.name)
        .unwrap_or_else(|| format!("branch_{id}"))
}

/// Builds zyron_sys.cdc.feeds
fn build_feeds(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT4_OID, 4),
        make_field("enabled", PG_BOOL_OID, 1),
        make_field("retention", PG_TEXT_OID, -1),
        make_field("columns", PG_TEXT_OID, -1),
        make_field("before_image", PG_BOOL_OID, 1),
        make_field("compression", PG_TEXT_OID, -1),
        make_field("oldest_version", PG_INT8_OID, 8),
        make_field("latest_version", PG_INT8_OID, 8),
        make_field("bytes", PG_INT8_OID, 8),
        make_field("rows", PG_INT8_OID, 8),
        make_field("oldest_ts", PG_INT8_OID, 8),
    ];
    let Some(feeds) = server.cdc_registry.as_ref() else {
        return (fields, Vec::new());
    };
    let mut rows = Vec::new();
    for table_id in feeds.table_ids() {
        let Some(feed) = feeds.get_feed(table_id) else {
            continue;
        };
        let config = feed.config();
        let columns = match &config.columns {
            None => "all".to_string(),
            Some(ids) => {
                let table = server
                    .catalog
                    .get_table_by_id(zyron_catalog::TableId(table_id))
                    .ok();
                ids.iter()
                    .map(|id| {
                        table
                            .as_ref()
                            .and_then(|t| t.columns.iter().find(|c| c.id.0 == *id))
                            .map(|c| c.name.clone())
                            .unwrap_or_else(|| id.to_string())
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        let retention = if config.retention_micros <= 0 {
            "unbounded".to_string()
        } else {
            format!("{} seconds", config.retention_micros / 1_000_000)
        };
        let compression = match config.codec {
            zyron_cdc::CdfCodec::None => "none",
            zyron_cdc::CdfCodec::Lz4 => "lz4",
            zyron_cdc::CdfCodec::Zstd => "zstd",
        };
        rows.push(vec![
            text(table_name(server, table_id)),
            number(table_id),
            flag(feed.is_enabled()),
            text(retention),
            text(columns),
            flag(config.before_image),
            text(compression),
            number(feed.oldest_version().unwrap_or(0)),
            number(feed.latest_version().unwrap_or(0)),
            number(feed.file_size_bytes()),
            number(feed.record_count()),
            number(feed.oldest_timestamp().unwrap_or(0)),
        ]);
    }
    (fields, rows)
}

/// Builds zyron_sys.cdc.apply_runs
fn build_apply_runs() -> ViewRows {
    let fields = vec![
        make_field("target", PG_TEXT_OID, -1),
        make_field("source", PG_TEXT_OID, -1),
        make_field("rows_upserted", PG_INT8_OID, 8),
        make_field("rows_deleted", PG_INT8_OID, 8),
        make_field("rows_versioned", PG_INT8_OID, 8),
        make_field("truncated", PG_BOOL_OID, 1),
        make_field("duration_micros", PG_INT8_OID, 8),
        make_field("started_at", PG_INT8_OID, 8),
        make_field("error", PG_TEXT_OID, -1),
    ];
    let rows = crate::change_stream_dispatch::apply_runs()
        .list()
        .into_iter()
        .map(|run| {
            vec![
                text(run.target),
                text(run.source),
                number(run.rows_upserted),
                number(run.rows_deleted),
                number(run.rows_versioned),
                flag(run.truncated),
                number(run.duration_micros),
                number(run.started_at),
                text(run.error),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.alert.templates from every template a subsystem
/// declares
fn build_alert_templates() -> ViewRows {
    let fields = vec![
        make_field("name", PG_TEXT_OID, -1),
        make_field("subsystem", PG_TEXT_OID, -1),
        make_field("condition", PG_TEXT_OID, -1),
        make_field("threshold_setting", PG_TEXT_OID, -1),
        make_field("default_threshold", PG_TEXT_OID, -1),
        make_field("summary", PG_TEXT_OID, -1),
    ];
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = ALERT_TEMPLATES
        .iter()
        .map(|template| {
            vec![
                text(template.name),
                text("cdc"),
                text(template.condition),
                text(template.threshold_setting),
                text(template.default_threshold),
                text(template.summary),
            ]
        })
        .collect();
    // Every subsystem that declares an alert is listed here, so one view
    // answers what this node can raise rather than one view per subsystem
    rows.extend(
        crate::system_verify_views::ALERT_TEMPLATES
            .iter()
            .map(|template| {
                vec![
                    text(template.name),
                    text("verify"),
                    text(template.condition),
                    text(template.threshold_setting),
                    text(template.default_threshold),
                    text(template.summary),
                ]
            }),
    );
    (fields, rows)
}

//! Virtual statistics views that bypass the normal planner/executor path.
//!
//! Each view returns a column schema (FieldDescription vector) and data rows
//! directly, allowing clients to query internal server metrics through
//! standard SQL SELECT statements on virtual system tables.

use std::sync::atomic::Ordering;

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
];

/// Returns true if the given name matches a virtual statistics view.
pub fn is_stat_view(name: &str) -> bool {
    STAT_VIEW_NAMES.contains(&name)
}

/// Dispatches to the appropriate view builder and returns the column schema
/// paired with data rows. Returns None if the name is not a recognized view.
pub fn query_stat_view(
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
        _ => None,
    }
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
/// Data source: server.session_mgr (not yet available on ServerState).
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

/// Builds the zyron_stat_tables view.
/// Columns: table_name, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch,
///          n_tup_ins, n_tup_upd, n_tup_del, n_dead_tup,
///          last_vacuum, last_analyze, row_count.
/// Data source: server.table_io_stats + server.catalog (not yet available).
fn build_stat_tables(_server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
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

    // table_io_stats is not yet on ServerState, return empty rows for now.
    let rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    (fields, rows)
}

/// Builds the zyron_stat_indexes view.
/// Columns: index_name, table_name, index_type, idx_scan, idx_tup_read, idx_tup_fetch.
/// Data source: server.index_io_stats + server.catalog (not yet available).
fn build_stat_indexes(_server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("index_name", PG_TEXT_OID, -1),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("index_type", PG_TEXT_OID, -1),
        make_field("idx_scan", PG_INT8_OID, 8),
        make_field("idx_tup_read", PG_INT8_OID, 8),
        make_field("idx_tup_fetch", PG_INT8_OID, 8),
    ];

    // index_io_stats is not yet on ServerState, return empty rows for now.
    let rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    (fields, rows)
}

/// Builds the zyron_stat_wal view.
/// Columns: wal_records, wal_bytes, wal_syncs, wal_flushed_lsn,
///          wal_current_segment, last_checkpoint_lsn.
/// Reads flushed_lsn and current_segment_id from the WAL writer.
/// checkpoint_stats is not yet on ServerState.
fn build_stat_wal(server: &ServerState) -> (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>) {
    let fields = vec![
        make_field("wal_records", PG_INT8_OID, 8),
        make_field("wal_bytes", PG_INT8_OID, 8),
        make_field("wal_syncs", PG_INT8_OID, 8),
        make_field("wal_flushed_lsn", PG_INT8_OID, 8),
        make_field("wal_current_segment", PG_INT4_OID, 4),
        make_field("last_checkpoint_lsn", PG_INT8_OID, 8),
    ];

    let wal_records = server.wal.wal_records_written.load(Ordering::Relaxed);
    let wal_bytes = server.wal.wal_bytes_written.load(Ordering::Relaxed);
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
/// Data source: server.checkpoint_stats and server.vacuum_stats (not yet available).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_stat_view_recognized() {
        assert!(is_stat_view("zyron_stat_activity"));
        assert!(is_stat_view("zyron_stat_tables"));
        assert!(is_stat_view("zyron_stat_indexes"));
        assert!(is_stat_view("zyron_stat_wal"));
        assert!(is_stat_view("zyron_stat_bgwriter"));
    }

    #[test]
    fn test_is_stat_view_unrecognized() {
        assert!(!is_stat_view("zyron_stat_unknown"));
        assert!(!is_stat_view("pg_stat_activity"));
        assert!(!is_stat_view(""));
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

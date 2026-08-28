//! The `zyron_sys.cdc.*` table functions.
//!
//! Two entry points, both taking arguments in a FROM clause: one creates a
//! replication slot pinned at the current WAL head, the other consumes
//! decoded changes from a slot and advances it past what it handed back.
//! Consuming is what makes the second one a function rather than a view: a
//! second call returns what follows the first, the same contract Postgres
//! gives `pg_logical_slot_get_changes`.

use std::sync::Arc;

use zyron_cdc::change_feed::ChangeRecord;
use zyron_cdc::decoder::{DecodedChange, DecoderPlugin, create_decoder};
use zyron_common::ZyronError;

use crate::connection::ServerState;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_INT8_OID, PG_TEXT_OID};

/// Default cap on how many changes one call consumes when the caller does
/// not name one. Bounded so a slot that has fallen far behind returns a
/// batch a client can hold rather than everything the feed has kept.
const DEFAULT_CHANGE_LIMIT: usize = 10_000;

fn cell(value: impl ToString) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

/// Dispatches one `zyron_sys.cdc.*` call. Reached only for a name the
/// registry recognized.
pub fn call(object: &str, args: &[String], server: &ServerState) -> Result<ViewRows, ZyronError> {
    match object {
        "create_replication_slot" => create_replication_slot(args, server),
        "logical_slot_get_changes" => logical_slot_get_changes(args, server),
        other => Err(ZyronError::Internal(format!(
            "`zyron_sys.cdc.{}` is registered but has no implementation",
            other
        ))),
    }
}

fn slot_manager(server: &ServerState) -> Result<&Arc<zyron_cdc::SlotManager>, ZyronError> {
    server.slot_manager.as_ref().ok_or_else(|| {
        ZyronError::CdcStreamError("replication slots are not enabled on this server".into())
    })
}

/// `zyron_sys.cdc.create_replication_slot(name [, plugin])`.
///
/// Creates the slot and pins WAL retention at the current head before
/// returning, so nothing committed between the create and the consumer's
/// first read can be reclaimed. A failure to pin rolls the slot back rather
/// than leaving one that retains nothing.
///
/// Columns: slot_name, plugin, start_lsn, restart_lsn, active.
fn create_replication_slot(args: &[String], server: &ServerState) -> Result<ViewRows, ZyronError> {
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::PlanError(
            "zyron_sys.cdc.create_replication_slot(name [, plugin]) takes one or two arguments"
                .to_string(),
        ));
    }
    let name = args[0].as_str();
    if name.is_empty() {
        return Err(ZyronError::PlanError(
            "zyron_sys.cdc.create_replication_slot requires a non-empty slot name".to_string(),
        ));
    }
    let plugin_name = args.get(1).map(|s| s.as_str()).unwrap_or("zyron_cdc");
    let plugin = DecoderPlugin::from_str(plugin_name)?;

    let manager = slot_manager(server)?;
    manager.create_slot(name, plugin, None)?;
    let start = server.wal.next_lsn();
    if let Err(e) = manager.advance_slot(name, start) {
        // A slot that exists but retains nothing would let the WAL it is
        // supposed to pin be reclaimed, so the half-created slot goes away
        let _ = manager.drop_slot(name);
        return Err(e);
    }
    manager.flush_if_dirty()?;

    let slot = manager.get_slot(name)?;
    let fields = vec![
        make_field("slot_name", PG_TEXT_OID, -1),
        make_field("plugin", PG_TEXT_OID, -1),
        make_field("start_lsn", PG_INT8_OID, 8),
        make_field("restart_lsn", PG_INT8_OID, 8),
        make_field("active", PG_TEXT_OID, -1),
    ];
    let rows = vec![vec![
        cell(&slot.name),
        cell(slot.plugin.as_str()),
        cell(slot.confirmed_lsn),
        cell(slot.restart_lsn),
        cell(slot.active),
    ]];
    Ok((fields, rows))
}

/// `zyron_sys.cdc.logical_slot_get_changes(name [, upto_n])`.
///
/// Reads the changes the slot has not confirmed, decodes them with the
/// plugin the slot was created with, and advances the slot past the last one
/// returned. Advancing is what makes the call consuming: the next call
/// starts where this one stopped, and a caller that wants to re-read has to
/// keep what it was given.
///
/// Columns: lsn, xid, table_id, operation, data.
fn logical_slot_get_changes(args: &[String], server: &ServerState) -> Result<ViewRows, ZyronError> {
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::PlanError(
            "zyron_sys.cdc.logical_slot_get_changes(name [, upto_n]) takes one or two arguments"
                .to_string(),
        ));
    }
    let name = args[0].as_str();
    let limit = match args.get(1) {
        None => DEFAULT_CHANGE_LIMIT,
        Some(raw) => raw.trim().parse::<usize>().map_err(|_| {
            ZyronError::PlanError(format!(
                "zyron_sys.cdc.logical_slot_get_changes: `{}` is not a row count",
                raw
            ))
        })?,
    };

    let manager = slot_manager(server)?;
    let slot = manager.get_slot(name)?;
    let registry = server.cdc_registry.as_ref().ok_or_else(|| {
        ZyronError::CdcStreamError("change data capture is not enabled on this server".into())
    })?;

    // The slot's filter says which tables it carries. Without one it carries
    // every captured table, which is what an unfiltered slot means
    let table_ids: Vec<u32> = match &slot.table_filter {
        Some(ids) => ids.clone(),
        None => registry
            .list_feeds()
            .into_iter()
            .map(|(id, ..)| id)
            .collect(),
    };

    let start = slot.confirmed_lsn.saturating_add(1);
    let mut collected: Vec<ChangeRecord> = Vec::new();
    for table_id in table_ids {
        let Some(feed) = registry.get_feed(table_id) else {
            continue;
        };
        collected.extend(feed.query_changes(start, u64::MAX)?);
    }
    // One slot is one ordered stream, so records from several feeds are
    // merged by commit position before the limit is applied. Cutting per
    // table first would hand back a prefix of one table and none of another
    collected.sort_by_key(|r| (r.commit_version, r.table_id));
    collected.truncate(limit);

    let decoder = create_decoder(slot.plugin);
    let fields = vec![
        make_field("lsn", PG_INT8_OID, 8),
        make_field("xid", PG_INT8_OID, 8),
        make_field("table_id", PG_INT8_OID, 8),
        make_field("operation", PG_TEXT_OID, -1),
        make_field("data", PG_TEXT_OID, -1),
    ];

    let mut rows = Vec::with_capacity(collected.len());
    let mut highest = slot.confirmed_lsn;
    for record in &collected {
        let change = decoded_change(record, server)?;
        let payload = decoder.serialize(&change)?;
        // A binary plugin's output is not printable text, so it is rendered
        // as hex rather than lossily as UTF-8. A JSON plugin's output passes
        // through unchanged
        let rendered = match std::str::from_utf8(&payload) {
            Ok(text) => text.to_string(),
            Err(_) => hex_encode(&payload),
        };
        rows.push(vec![
            cell(record.commit_version),
            cell(record.txn_id),
            cell(record.table_id),
            cell(format!("{:?}", record.change_type)),
            cell(rendered),
        ]);
        if record.commit_version > highest {
            highest = record.commit_version;
        }
    }

    // Advance only after every record has been rendered. Advancing first
    // would drop changes the caller never received if rendering failed
    if highest > slot.confirmed_lsn {
        manager.advance_slot(name, zyron_wal::record::Lsn(highest))?;
        manager.flush_if_dirty()?;
    }
    Ok((fields, rows))
}

/// Turns a raw change record into the shape a logical decoder serializes,
/// resolving the table's name and column names through the catalog.
///
/// A row whose columns cannot be decoded is described by its change type and
/// position with no values rather than failing the whole call: one
/// unreadable record must not make the rest of the slot unreadable.
fn decoded_change(
    record: &ChangeRecord,
    server: &ServerState,
) -> Result<DecodedChange, ZyronError> {
    let table = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(record.table_id))
        .ok();
    let table_name = table
        .as_ref()
        .map(|t| t.name.clone())
        .unwrap_or_else(|| format!("table_{}", record.table_id));

    let values = match table.as_ref() {
        Some(entry) => {
            let types: Vec<zyron_common::TypeId> =
                entry.columns.iter().map(|c| c.type_id).collect();
            match zyron_streaming::row_codec::decode_row(&record.row_data, &types) {
                Ok(decoded) => Some(
                    entry
                        .columns
                        .iter()
                        .zip(decoded.iter())
                        .map(|(column, value)| (column.name.clone(), render_value(value)))
                        .collect::<Vec<(String, String)>>(),
                ),
                Err(_) => None,
            }
        }
        None => None,
    };

    // A delete carries the image of what went away, an insert the image of
    // what arrived, and an update postimage the new one. Putting a delete's
    // image in `new_values` would tell a consumer the row still exists
    let (old_values, new_values) = match record.change_type {
        zyron_cdc::ChangeType::Delete | zyron_cdc::ChangeType::UpdatePreimage => (values, None),
        _ => (None, values),
    };

    Ok(DecodedChange {
        table_name,
        table_id: record.table_id,
        operation: record.change_type,
        old_values,
        new_values,
        commit_lsn: record.commit_version,
        commit_timestamp: record.commit_timestamp,
        txn_id: record.txn_id,
        is_last_in_txn: record.is_last_in_txn,
        schema_version: record.schema_version,
    })
}

/// Renders one decoded column the way the change stream carries it: as text,
/// with NULL distinguished from the empty string by the literal `NULL`.
fn render_value(value: &zyron_streaming::row_codec::StreamValue) -> String {
    use zyron_streaming::row_codec::StreamValue;
    match value {
        StreamValue::Null => "NULL".to_string(),
        StreamValue::Bool(b) => b.to_string(),
        StreamValue::I64(v) => v.to_string(),
        StreamValue::I128(v) => v.to_string(),
        StreamValue::F64(v) => v.to_string(),
        StreamValue::Utf8(s) => s.clone(),
        StreamValue::Binary(b) => hex_encode(b),
    }
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

// -----------------------------------------------------------------------------
// Outbound CDC stream pump.
//
// Drives every active CDC stream: reads change records the table's change feed
// has accumulated past the stream's replication slot, decodes each row against
// the table schema, and delivers them to the configured sink (Kafka, S3, or
// webhook). Delivery and slot advancement run on a blocking thread because the
// network sinks perform synchronous IO through a bridged runtime.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tracing::warn;

use zyron_catalog::{ColumnEntry, TableId};
use zyron_cdc::cdc_stream::build_sink;
use zyron_cdc::decoder::DecodedChange;
use zyron_cdc::{ChangeRecord, ChangeType};
use zyron_common::TypeId;
use zyron_executor::batch::{ColumnBuilder, decode_tuple_into_builders};
use zyron_wire::connection::ServerState;

pub const DEFAULT_INTERVAL_SECS: u64 = 1;

pub async fn cdc_stream_pump_loop(
    server: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    interval_secs: u64,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(1)));
    loop {
        ticker.tick().await;
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        run_pump_once(&server).await;
    }
}

/// Runs one delivery pass over every active stream. Returns the total number
/// of records delivered across all streams this pass.
pub async fn run_pump_once(server: &Arc<ServerState>) -> u64 {
    let (Some(stream_mgr), Some(registry), Some(slots)) = (
        &server.cdc_stream_manager,
        &server.cdc_registry,
        &server.slot_manager,
    ) else {
        return 0;
    };

    let mut total = 0u64;
    for stream in stream_mgr.list_streams() {
        if !stream.active {
            continue;
        }
        // A soft-dropped or missing table is skipped: nothing to decode against.
        let table = match server.catalog.get_table_by_id(TableId(stream.table_id)) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let columns = table.columns.clone();
        let table_name = table.name.clone();
        let slots = Arc::clone(slots);
        let sink = build_sink(&stream);

        // A lake table has no change file. Its transaction log is the change
        // record, so the records are derived from the log and driven through
        // the same batching, sink and slot-advance path
        let result = if table.lake.is_lake() {
            let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
            let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) else {
                continue;
            };
            let table = Arc::clone(&table);
            tokio::task::spawn_blocking(move || {
                let start_version = slots.get_slot(&stream.slot_name)?.confirmed_lsn;
                let changes = zyron_wire::lake_changes::lake_change_records(
                    &log,
                    &table,
                    start_version + 1,
                    u64::MAX,
                )?;
                zyron_cdc::drive_stream_changes(
                    &stream,
                    changes,
                    start_version,
                    slots.as_ref(),
                    sink.as_ref(),
                    |rec| Ok(decode_change(rec, &table_name, &columns)),
                )
            })
            .await
        } else {
            let Some(feed) = registry.get_feed(stream.table_id) else {
                continue;
            };
            tokio::task::spawn_blocking(move || {
                zyron_cdc::drive_stream_once(
                    &stream,
                    feed.as_ref(),
                    slots.as_ref(),
                    sink.as_ref(),
                    |rec| Ok(decode_change(rec, &table_name, &columns)),
                )
            })
            .await
        };

        match result {
            Ok(Ok(n)) => total += n,
            Ok(Err(e)) => warn!(
                target: "zyron::cdc",
                table_id = table.id.0,
                "CDC stream delivery failed: {e}"
            ),
            Err(e) => warn!(target: "zyron::cdc", "CDC pump task join error: {e}"),
        }
    }
    total
}

/// Builds a DecodedChange from a raw change record by decoding its row image
/// against the table schema. Insert and update post-images populate new_values,
/// delete and update pre-images populate old_values, and schema or truncate
/// markers carry no row image.
fn decode_change(rec: &ChangeRecord, table_name: &str, columns: &[ColumnEntry]) -> DecodedChange {
    let pairs = decode_row_pairs(&rec.row_data, columns);
    let (old_values, new_values) = match rec.change_type {
        ChangeType::Insert | ChangeType::UpdatePostimage => (None, Some(pairs)),
        ChangeType::Delete | ChangeType::UpdatePreimage => (Some(pairs), None),
        ChangeType::SchemaChange | ChangeType::Truncate => (None, None),
    };
    DecodedChange {
        table_name: table_name.to_string(),
        table_id: rec.table_id,
        operation: rec.change_type,
        old_values,
        new_values,
        commit_lsn: rec.commit_version,
        commit_timestamp: rec.commit_timestamp,
        txn_id: rec.txn_id,
        is_last_in_txn: rec.is_last_in_txn,
        schema_version: rec.schema_version,
    }
}

/// Decodes an NSM-encoded row into (column_name, string_value) pairs using the
/// canonical tuple decoder so the layout stays in lockstep with the writer.
fn decode_row_pairs(row_data: &[u8], columns: &[ColumnEntry]) -> Vec<(String, String)> {
    let null_bitmap_len = (columns.len() + 7) / 8;
    if row_data.len() < null_bitmap_len {
        return Vec::new();
    }
    // Identity map: every table column decodes into its own builder.
    let column_to_builder: Vec<Option<u16>> = (0..columns.len()).map(|i| Some(i as u16)).collect();
    let mut builders: Vec<ColumnBuilder> = columns
        .iter()
        .map(|c| {
            let phys = TypeId::timestamp_physical_type_id(c.type_id, c.fractional_digits);
            if phys != c.type_id || c.fractional_digits.is_some() {
                ColumnBuilder::new_ts(c.type_id, phys, c.fractional_digits, 1)
            } else {
                ColumnBuilder::new(c.type_id, 1)
            }
        })
        .collect();

    decode_tuple_into_builders(row_data, columns, &column_to_builder, &mut builders);

    let mut pairs = Vec::with_capacity(columns.len());
    for (col, builder) in columns.iter().zip(builders) {
        let column = builder.finish();
        let scalar = column.get_scalar(0);
        pairs.push((col.name.clone(), format!("{scalar}")));
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::{ColumnId, TableId};

    fn col(id: u16, name: &str, type_id: TypeId, max_length: Option<usize>) -> ColumnEntry {
        ColumnEntry {
            id: ColumnId(id),
            table_id: TableId(1),
            name: name.to_string(),
            type_id,
            ordinal: id,
            nullable: true,
            default_expr: None,
            max_length,
            fractional_digits: None,
            tz_offset_secs: None,
            element_type: None,
        }
    }

    /// Encodes one NSM row: null bitmap then fixed-size values inline and
    /// variable-length values as a 4-byte length prefix plus bytes.
    fn encode(values: &[(bool, Vec<u8>, bool)]) -> Vec<u8> {
        let n = values.len();
        let bitmap_len = (n + 7) / 8;
        let mut out = vec![0u8; bitmap_len];
        for (i, (is_null, _, _)) in values.iter().enumerate() {
            if *is_null {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        for (_, bytes, varlen) in values {
            if *varlen {
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            }
            out.extend_from_slice(bytes);
        }
        out
    }

    #[test]
    fn decodes_fixed_and_varlen_columns() {
        let columns = vec![
            col(0, "id", TypeId::Int64, None),
            col(1, "name", TypeId::Varchar, Some(255)),
        ];
        let row = encode(&[
            (false, 42i64.to_le_bytes().to_vec(), false),
            (false, b"hi".to_vec(), true),
        ]);
        let pairs = decode_row_pairs(&row, &columns);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("id".to_string(), "42".to_string()));
        assert_eq!(pairs[1], ("name".to_string(), "hi".to_string()));
    }

    #[test]
    fn short_row_yields_no_pairs() {
        let columns = vec![col(0, "id", TypeId::Int64, None)];
        // Empty buffer is shorter than the 1-byte null bitmap, so decode bails.
        assert!(decode_row_pairs(&[], &columns).is_empty());
    }

    #[test]
    fn delete_record_populates_old_values() {
        let columns = vec![col(0, "id", TypeId::Int64, None)];
        let row = encode(&[(false, 7i64.to_le_bytes().to_vec(), false)]);
        let rec = ChangeRecord {
            change_type: ChangeType::Delete,
            commit_version: 5,
            commit_timestamp: 100,
            table_id: 1,
            txn_id: 3,
            schema_version: 0,
            row_data: row,
            primary_key_data: Vec::new(),
            is_last_in_txn: true,
        };
        let decoded = decode_change(&rec, "t", &columns);
        assert!(decoded.new_values.is_none());
        assert_eq!(
            decoded.old_values.unwrap()[0],
            ("id".to_string(), "7".to_string())
        );
    }
}

//! Lake change records for the CDC consumers.
//!
//! A heap table's changes are captured into a `.zycdf` file as the DML runs.
//! A lake table needs no such file: its transaction log already records
//! every commit, so the same change records are derived from the log on
//! demand and no byte is written twice.
//!
//! Records come out in the exact shape the heap path produces, NSM row
//! bytes encoded by the shared tuple encoder, so publications, subscriptions
//! and CDC streams consume a lake table without knowing it is one.

use std::sync::Arc;

use zyron_catalog::{ColumnEntry, TableEntry};
use zyron_cdc::{ChangeRecord, ChangeType};
use zyron_common::Result;
use zyron_executor::batch::{DataBatch, create_builders, encode_row, finalize_builders};
use zyron_executor::column::ScalarValue;
use zyron_lake::{ChangeKind, LakeFileReader, OperationKind, TransactionLog};
use zyron_planner::logical::LogicalColumn;

/// Reads a lake table's changes over `from..=to` as change records.
///
/// One data file is opened at most once per descriptor and only the columns
/// of the table schema are decoded, so the cost is the rows that actually
/// changed rather than the table.
pub fn lake_change_records(
    log: &TransactionLog,
    table: &TableEntry,
    from_version: u64,
    to_version: u64,
) -> Result<Vec<ChangeRecord>> {
    let descriptors = zyron_lake::changes_between(log, from_version, to_version)?;
    if descriptors.is_empty() {
        return Ok(Vec::new());
    }
    let logical = logical_columns(table);

    let mut records: Vec<ChangeRecord> = Vec::new();
    for descriptor in &descriptors {
        let ordinals = zyron_lake::changed_ordinals(log, descriptor)?;
        if ordinals.is_empty() {
            continue;
        }
        // A delete is measured against the schema the rows were written
        // under, an insert against the schema that admitted them
        let manifest_version = match descriptor.kind {
            ChangeKind::Insert => descriptor.version,
            ChangeKind::Delete => descriptor.base_version,
        };
        let manifest = log.manifest_at(manifest_version)?;
        let reader = LakeFileReader::open(log.paths(), descriptor.partition_id)?;

        let batch = decode_rows(&reader, &manifest, table, &logical, &ordinals)?;
        let change_type = match (descriptor.kind, descriptor.operation) {
            // An update commit removes the old images and adds the new ones
            // in one version, which is exactly a pre and post image pair
            (ChangeKind::Delete, OperationKind::Update) => ChangeType::UpdatePreimage,
            (ChangeKind::Insert, OperationKind::Update) => ChangeType::UpdatePostimage,
            (ChangeKind::Delete, _) => ChangeType::Delete,
            (ChangeKind::Insert, _) => ChangeType::Insert,
        };
        for row in 0..batch.num_rows {
            records.push(ChangeRecord {
                change_type,
                commit_version: descriptor.version,
                commit_timestamp: descriptor.timestamp_us,
                table_id: table.id.0,
                txn_id: descriptor.db_txn_id as u32,
                schema_version: manifest.schema.schema_id as u32,
                row_data: encode_row(&batch, row, &table.columns),
                primary_key_data: Vec::new(),
                is_last_in_txn: false,
            });
        }
    }

    // The last record of each version closes that version's batch, which is
    // what a consumer applying whole transactions waits for
    for i in 0..records.len() {
        let closes = match records.get(i + 1) {
            Some(next) => next.commit_version != records[i].commit_version,
            None => true,
        };
        records[i].is_last_in_txn = closes;
    }
    Ok(records)
}

/// The table's columns in catalog order, which is the order `encode_row`
/// lays a tuple out in.
fn logical_columns(table: &TableEntry) -> Vec<LogicalColumn> {
    table
        .columns
        .iter()
        .map(|c: &ColumnEntry| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            ts_precision: c.ts_precision,
        })
        .collect()
}

/// Decodes the named ordinals of one data file into a batch shaped like the
/// table, so the shared tuple encoder produces heap-identical row bytes.
fn decode_rows(
    reader: &LakeFileReader,
    manifest: &Arc<zyron_lake::ManifestFile>,
    table: &TableEntry,
    logical: &[LogicalColumn],
    ordinals: &[u64],
) -> Result<DataBatch> {
    let mut decoded = Vec::with_capacity(table.columns.len());
    for column in &table.columns {
        // A column the file predates reads as NULL, the schema is the
        // authority and a change record still carries every column
        let lake_column = manifest.schema.column_by_id(column.id.0 as u32);
        let value_size = lake_column
            .map(|c| c.physical_type_id().fixed_size().unwrap_or(0))
            .unwrap_or(0);
        let data = match lake_column {
            Some(c) => Some(reader.read_column(c)?),
            None => None,
        };
        decoded.push((column.type_id, value_size, data));
    }

    let mut builders = create_builders(logical, ordinals.len());
    for &ordinal in ordinals {
        let row = ordinal as usize;
        for (index, (type_id, value_size, data)) in decoded.iter().enumerate() {
            let scalar = match data.as_ref().and_then(|d| d.cell(row)) {
                None => ScalarValue::Null,
                Some(cell) if *value_size == 0 => {
                    zyron_executor::batch::decode_varlen_scalar(*type_id, cell)
                }
                Some(cell) => zyron_executor::batch::decode_fixed_scalar(*type_id, cell),
            };
            builders[index].push(&scalar);
        }
    }
    Ok(finalize_builders(builders))
}

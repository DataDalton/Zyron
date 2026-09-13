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
//!
//! The rows of one derivation are encoded into one buffer, each record a
//! span of it, so a read of ten million lake changes allocates a buffer per
//! commit rather than a row per change

use std::sync::Arc;

use zyron_catalog::{ColumnEntry, TableEntry};
use zyron_cdc::{ChangeRecord, ChangeType};
use zyron_common::Result;
use zyron_executor::batch::{DataBatch, create_builders, encode_row_into, finalize_builders};
use zyron_executor::column::ScalarValue;
use zyron_lake::{ChangeKind, LakeFileReader, OperationKind, TransactionLog};
use zyron_planner::logical::LogicalColumn;

/// What one derived change carries beside its row bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LakeChangeHead {
    change_type: ChangeType,
    commit_version: u64,
    commit_timestamp: i64,
    txn_id: u64,
    change_ordinal: u64,
    schema_version: u32,
    is_last_in_txn: bool,
    /// Where the row's bytes start in the shared buffer, ending where the
    /// next record's start or the buffer ends
    start: usize,
}

/// The changes of one derivation, their row bytes in one buffer
#[derive(Debug, Default)]
pub struct LakeChangeRows {
    table_id: u32,
    heads: Vec<LakeChangeHead>,
    rows: Vec<u8>,
}

/// One derived change, its row bytes borrowed from the derivation's buffer
#[derive(Debug, Clone, Copy)]
pub struct LakeChangeRow<'a> {
    pub change_type: ChangeType,
    pub commit_version: u64,
    pub commit_timestamp: i64,
    pub table_id: u32,
    pub txn_id: u64,
    pub change_ordinal: u64,
    pub schema_version: u32,
    pub is_last_in_txn: bool,
    pub row_data: &'a [u8],
}

impl LakeChangeRows {
    /// How many changes the derivation holds
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// The change at `at`, in commit version then position order
    pub fn row(&self, at: usize) -> LakeChangeRow<'_> {
        let head = self.heads[at];
        let end = self
            .heads
            .get(at + 1)
            .map(|next| next.start)
            .unwrap_or(self.rows.len());
        LakeChangeRow {
            change_type: head.change_type,
            commit_version: head.commit_version,
            commit_timestamp: head.commit_timestamp,
            table_id: self.table_id,
            txn_id: head.txn_id,
            change_ordinal: head.change_ordinal,
            schema_version: head.schema_version,
            is_last_in_txn: head.is_last_in_txn,
            row_data: &self.rows[head.start..end],
        }
    }

    /// The changes as owned records, for a consumer that keeps them
    pub fn into_records(self) -> Vec<ChangeRecord> {
        (0..self.len())
            .map(|at| {
                let row = self.row(at);
                ChangeRecord {
                    change_type: row.change_type,
                    commit_version: row.commit_version,
                    commit_timestamp: row.commit_timestamp,
                    table_id: row.table_id,
                    txn_id: row.txn_id,
                    change_ordinal: row.change_ordinal,
                    schema_version: row.schema_version,
                    row_data: row.row_data.to_vec(),
                    primary_key_data: Vec::new(),
                    is_last_in_txn: row.is_last_in_txn,
                    projected: false,
                }
            })
            .collect()
    }
}

/// Reads a lake table's changes over `from..=to` as change records.
///
/// The owned form of [`lake_change_rows`], for a consumer that keeps the
/// records
pub fn lake_change_records(
    log: &TransactionLog,
    table: &TableEntry,
    from_version: u64,
    to_version: u64,
    before_images: bool,
) -> Result<Vec<ChangeRecord>> {
    Ok(lake_change_rows(log, table, from_version, to_version, before_images)?.into_records())
}

/// Reads a lake table's changes over `from..=to`, their rows in one buffer.
///
/// One data file is opened at most once per descriptor and only the columns
/// of the table schema are decoded, so the cost is the rows that actually
/// changed rather than the table. `before_images` says whether an update's
/// removed side is a record, the feed's setting for these versions
pub fn lake_change_rows(
    log: &TransactionLog,
    table: &TableEntry,
    from_version: u64,
    to_version: u64,
    before_images: bool,
) -> Result<LakeChangeRows> {
    let descriptors = zyron_lake::changes_between(log, from_version, to_version)?;
    let mut out = LakeChangeRows {
        table_id: table.id.0,
        heads: Vec::new(),
        rows: Vec::new(),
    };
    if descriptors.is_empty() {
        return Ok(out);
    }
    let logical = logical_columns(table);

    // A truncate is one record for the whole table, whatever the commit
    // removed, so the first descriptor of a truncating version records it
    // and the rest of that version's are passed over
    let mut truncated_at: Option<u64> = None;
    for descriptor in &descriptors {
        // The transaction a reader's snapshot judges the record by, which
        // for a commit made under a cross table intent is the transaction
        // that opened the intent
        let txn_id = zyron_lake::owning_txn(log.paths(), descriptor.db_txn_id);
        if descriptor.truncate {
            if truncated_at != Some(descriptor.version) {
                truncated_at = Some(descriptor.version);
                let start = out.rows.len();
                out.heads.push(LakeChangeHead {
                    change_type: ChangeType::Truncate,
                    commit_version: descriptor.version,
                    commit_timestamp: descriptor.timestamp_us,
                    txn_id,
                    change_ordinal: 0,
                    schema_version: table.schema_epoch as u32,
                    is_last_in_txn: false,
                    start,
                });
            }
            continue;
        }
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
        let reader = LakeFileReader::open_in(&manifest, log.paths(), descriptor.partition_id)?;

        let change_type = match (descriptor.kind, descriptor.operation) {
            // An update commit removes the old images and adds the new ones
            // in one version, which is exactly a pre and post image pair
            (ChangeKind::Delete, OperationKind::Update) => ChangeType::UpdatePreimage,
            (ChangeKind::Insert, OperationKind::Update) => ChangeType::UpdatePostimage,
            (ChangeKind::Delete, _) => ChangeType::Delete,
            (ChangeKind::Insert, _) => ChangeType::Insert,
        };
        // A feed keeping one image of an update reports the postimage alone
        if change_type == ChangeType::UpdatePreimage && !before_images {
            continue;
        }
        let batch = decode_rows(&reader, &manifest, table, &logical, &ordinals)?;
        out.heads.reserve(batch.num_rows);
        for row in 0..batch.num_rows {
            let start = out.rows.len();
            // The row is laid out over the table's column list as it stands,
            // which is the layout its current schema epoch names, so that is
            // the epoch the record carries whatever lake schema the file was
            // written under
            encode_row_into(&mut out.rows, &batch, row, &table.columns);
            out.heads.push(LakeChangeHead {
                change_type,
                commit_version: descriptor.version,
                commit_timestamp: descriptor.timestamp_us,
                txn_id,
                change_ordinal: 0,
                schema_version: table.schema_epoch as u32,
                is_last_in_txn: false,
                start,
            });
        }
    }

    // A lake table has no change file to number records in as they are
    // written, so the position inside a commit is assigned here, over the
    // records the commit produced in log entry order. An update's delete side
    // therefore precedes its insert side under one version, the same pairing
    // a heap feed records
    let mut ordinal = 0u64;
    for i in 0..out.heads.len() {
        if i > 0 && out.heads[i].commit_version != out.heads[i - 1].commit_version {
            ordinal = 0;
        }
        out.heads[i].change_ordinal = ordinal;
        ordinal += 1;
    }

    // The last record of each version closes that version's batch, which is
    // what a consumer applying whole transactions waits for
    for i in 0..out.heads.len() {
        let closes = match out.heads.get(i + 1) {
            Some(next) => next.commit_version != out.heads[i].commit_version,
            None => true,
        };
        out.heads[i].is_last_in_txn = closes;
    }
    Ok(out)
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
            fractional_digits: c.fractional_digits,
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
        // authority and a change record still carries every column. A
        // dropped column is out of the schema from the drop on, so a change
        // committed after it reads NULL there whether or not the file still
        // holds the segment, the same NULL a heap image taken after the
        // drop carries, while a change committed before the drop reads
        // through the schema of its own version and keeps the value
        // The cells are read in the shape the table's column declares now,
        // which is the layout the record's row image follows, so a file
        // written while the column was narrower hands its cells over
        // widened and a version read from before the column widened does
        // the same
        let lake_column = manifest.schema.column_by_id(column.id.0 as u32).map(|c| {
            zyron_executor::operator::lake_scan::declared_as(
                c,
                column.type_id,
                column.fractional_digits,
            )
        });
        let value_size = lake_column
            .as_ref()
            .map(|c| c.physical_type_id().fixed_size().unwrap_or(0))
            .unwrap_or(0);
        let data = match &lake_column {
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

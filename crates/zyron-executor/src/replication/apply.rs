//! Putting another node's committed transaction back.
//!
//! ## The same paths that produced the rows
//!
//! Apply does not reach into heap pages. It builds the same operators an
//! ordinary statement would, over rows it decodes rather than rows it
//! computes, so every storage kind, every index kind and every tier is handled
//! by the code that is already exercised by every other test in the engine.
//! A second hand-written write path would be a second thing to keep correct as
//! the first one changes.
//!
//! What is turned off is only the deciding: constraints, foreign keys, checks
//! and triggers all ran on the leader, and the group has agreed on their
//! outcome. Index maintenance, the WAL and the change feed stay on, because
//! those describe what happened rather than deciding it. The gate is
//! [`crate::context::ExecutionContext::replication_apply`].
//!
//! ## Finding the rows a delete names
//!
//! With a replica identity the changeset carries an encoded index key and this
//! probes the index. Without one it carries whole row images and this scans,
//! matching encoded bytes. The bytes match exactly rather than approximately,
//! because the row on this node was written from the same encoding the leader
//! shipped, so equality of rows is equality of bytes.
//!
//! An update is applied as the delete of its old image followed by the insert
//! of its new one. A row's identity here is its key or its contents, never the
//! slot it occupies, so the visible result is the same and there is one
//! implementation of index maintenance rather than two.

use std::sync::Arc;

use hashbrown::HashMap;
use zyron_catalog::{TableEntry, TableId};
use zyron_common::{Result, RowLocator, ZyronError};
use zyron_planner::logical::LogicalColumn;

use crate::batch::{DataBatch, create_builders, encode_row_into, finalize_builders};
use crate::context::ExecutionContext;
use crate::operator::modify::{DeleteOperator, InsertOperator, index_key_upper_bound};
use crate::operator::scan::{IndexScanOperator, SeqScanOperator};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

use super::changeset::{ChangesetOp, RowImage};

/// Hands an already built batch to whatever consumes it.
///
/// An insert needs a source operator and the rows are already decoded, so this
/// is the whole of it
struct PreparedBatchOperator {
    batch: Option<DataBatch>,
}

impl Operator for PreparedBatchOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.batch.take().map(ExecutionBatch::new)) })
    }
}

/// Every column of a table, in declaration order, as a scan describes them.
fn all_columns(table: &TableEntry) -> Vec<LogicalColumn> {
    table
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect()
}

/// Refuses a changeset written against a different shape of table.
///
/// The schema this replays against is the leader's, because a schema change is
/// a barrier in the same log and everything after it is ordered behind it. A
/// mismatch is therefore a defect rather than a race, and it has to stop the
/// node instead of decoding a row against the wrong columns
fn check_shape(table: &TableEntry, columns: u16) -> Result<()> {
    if table.columns.len() != columns as usize {
        return Err(ZyronError::RaftLogCorrupted {
            index: 0,
            reason: format!(
                "changeset names table {} with {} columns, this node holds {}",
                table.name,
                columns,
                table.columns.len()
            ),
        });
    }
    Ok(())
}

/// Turns encoded row images back into a batch.
///
/// The images are slices of the log record this node already holds, so the
/// only copy is the one into the column buffers
fn decode_rows(table: &TableEntry, rows: &[&[u8]]) -> Result<DataBatch> {
    let columns = all_columns(table);
    let output_ids: Vec<zyron_catalog::ColumnId> = columns.iter().map(|c| c.column_id).collect();
    let decoder = crate::epoch_decode::EpochDecoder::new(table, &output_ids);
    let mut builders = create_builders(&columns, rows.len());
    // A schema change is a barrier in the changeset, so every row that
    // follows it was encoded by the leader against the schema this node has
    // already applied. The epoch the leader wrote under is therefore the one
    // this table carries now
    let epoch = table.schema_epoch;
    for row in rows {
        decoder.decode(epoch, row, None, &mut builders)?;
    }
    Ok(finalize_builders(builders))
}

/// Runs an operator to exhaustion, which is what a DML operator being driven
/// looks like when nobody wants the row count it reports
async fn drain(mut op: Box<dyn Operator>) -> Result<()> {
    while op.next().await?.is_some() {}
    Ok(())
}

/// Applies rows entering a table.
pub async fn apply_insert(
    ctx: &Arc<ExecutionContext>,
    table_id: u32,
    columns: u16,
    rows: &[&[u8]],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let table = ctx.get_table_entry(TableId(table_id))?;
    check_shape(&table, columns)?;
    let batch = decode_rows(&table, rows)?;
    let target: Vec<zyron_catalog::ColumnId> = table.columns.iter().map(|c| c.id).collect();
    let source = Box::new(PreparedBatchOperator { batch: Some(batch) }) as Box<dyn Operator>;
    // No defaults, no checks, no expectations: the row image is what the
    // leader stored after applying all three
    let op = InsertOperator::new(
        source,
        Arc::clone(ctx),
        TableId(table_id),
        target,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        // Replicated rows carry the leader's computed stored generation
        Vec::new(),
    );
    drain(Box::new(op)).await
}

/// Applies rows leaving a table.
pub async fn apply_delete(
    ctx: &Arc<ExecutionContext>,
    table_id: u32,
    columns: u16,
    index_id: Option<u32>,
    rows: &[RowImage<'_>],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let table = ctx.get_table_entry(TableId(table_id))?;
    check_shape(&table, columns)?;
    let locators = resolve(ctx, &table, index_id, rows).await?;
    if locators.is_empty() {
        return Ok(());
    }
    delete_locators(ctx, &table, locators).await
}

/// Applies rows changing, as a delete of the old image and an insert of the
/// new one under the same transaction.
pub async fn apply_update(
    ctx: &Arc<ExecutionContext>,
    table_id: u32,
    columns: u16,
    index_id: Option<u32>,
    rows: &[(&[u8], &[u8])],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let table = ctx.get_table_entry(TableId(table_id))?;
    check_shape(&table, columns)?;

    let old: Vec<RowImage<'_>> = rows
        .iter()
        .map(|(old, _)| RowImage {
            bytes: old,
            multiplicity: 1,
        })
        .collect();
    let locators = resolve(ctx, &table, index_id, &old).await?;
    delete_locators(ctx, &table, locators).await?;

    let new: Vec<&[u8]> = rows.iter().map(|(_, new)| *new).collect();
    apply_insert(ctx, table_id, columns, &new).await
}

/// Drives a delete over rows the caller has already identified.
async fn delete_locators(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    locators: Vec<RowLocator>,
) -> Result<()> {
    let child =
        IndexScanOperator::from_locators(Arc::clone(ctx), table.id, all_columns(table), locators)
            .await?;
    let op = DeleteOperator::new(Box::new(child), Arc::clone(ctx), table.id);
    drain(Box::new(op)).await
}

/// Finds the rows a set of images names.
async fn resolve(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    index_id: Option<u32>,
    rows: &[RowImage<'_>],
) -> Result<Vec<RowLocator>> {
    match index_id {
        Some(index_id) => resolve_by_key(ctx, table, index_id, rows),
        None => resolve_by_image(ctx, table, rows).await,
    }
}

/// Probes the identity index once per distinct key.
///
/// Every locator the probe returns is passed on, including entries for rows
/// this snapshot cannot see: a deleted row keeps its index entry until vacuum
/// takes it, and the fetch that follows drops anything invisible. Filtering
/// here as well would mean a second visibility rule to keep in step with the
/// first
fn resolve_by_key(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    index_id: u32,
    rows: &[RowImage<'_>],
) -> Result<Vec<RowLocator>> {
    let Some(index) = ctx.get_index(zyron_catalog::IndexId(index_id)) else {
        return Err(ZyronError::RaftLogCorrupted {
            index: 0,
            reason: format!(
                "changeset names index {index_id} on table {}, which this node does not hold",
                table.name
            ),
        });
    };
    let heap_file_id = table.heap_file_id;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let before = out.len();
        let upper = index_key_upper_bound(row.bytes);
        index.range_scan_for_each(Some(row.bytes), upper.as_deref(), |_key, loc| {
            match loc {
                // A heap entry stores no file id, so it is re-stamped with the
                // table's the same way an index scan does
                RowLocator::Heap { page, slot } => out.push(RowLocator::Heap {
                    page: zyron_common::page::PageId::new(heap_file_id, page.page_num),
                    slot,
                }),
                other => out.push(other),
            }
            true
        });
        if out.len() == before {
            // The group agreed this row exists. Finding no entry at all for it
            // means this node holds a different table, which is worth stopping
            // for rather than applying around
            return Err(ZyronError::RaftLogCorrupted {
                index: 0,
                reason: format!(
                    "a changeset named a row of table {} that index {index_id} does not hold",
                    table.name
                ),
            });
        }
    }
    Ok(out)
}

/// Scans the table and matches whole encoded rows.
///
/// One pass covers every image in the operation, so a statement that deleted
/// a thousand rows from an index-less table costs one scan here just as it
/// cost one scan on the leader. Multiplicity is honoured exactly: a table
/// without a unique index can hold rows that are byte for byte the same, and
/// deleting one of them must delete one of them
async fn resolve_by_image(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    rows: &[RowImage<'_>],
) -> Result<Vec<RowLocator>> {
    let mut wanted: HashMap<&[u8], u32> = HashMap::with_capacity(rows.len());
    let mut total = 0u64;
    for row in rows {
        *wanted.entry(row.bytes).or_insert(0) += row.multiplicity;
        total += u64::from(row.multiplicity);
    }

    let columns = all_columns(table);
    let mut op: Box<dyn Operator> =
        Box::new(SeqScanOperator::new(Arc::clone(ctx), table.id, columns, None, true, None).await?);
    let mut out = Vec::with_capacity(total as usize);
    let mut found = 0u64;
    // One scratch buffer for every row the scan inspects, because encoding a
    // row is how it is compared and a vector per row would be an allocation
    // per row of the table
    let mut image: Vec<u8> = Vec::with_capacity(128);
    while let Some(batch) = op.next().await? {
        let Some(locators) = batch.locators.as_ref() else {
            return Err(ZyronError::Internal(
                "a replication match scan produced rows without locators".into(),
            ));
        };
        for row in 0..batch.batch.num_rows {
            if found == total {
                break;
            }
            image.clear();
            encode_row_into(&mut image, &batch.batch, row, &table.columns);
            let Some(left) = wanted.get_mut(image.as_slice()) else {
                continue;
            };
            if *left == 0 {
                continue;
            }
            *left -= 1;
            found += 1;
            out.push(locators[row]);
        }
        if found == total {
            break;
        }
    }
    if found != total {
        return Err(ZyronError::RaftLogCorrupted {
            index: 0,
            reason: format!(
                "a changeset named {total} rows of table {} and {found} of them are here",
                table.name
            ),
        });
    }
    Ok(out)
}

/// Applies one operation from a changeset.
pub async fn apply_op(ctx: &Arc<ExecutionContext>, op: &ChangesetOp<'_>) -> Result<()> {
    match op {
        ChangesetOp::Insert {
            table_id,
            columns,
            rows,
        } => apply_insert(ctx, *table_id, *columns, rows).await,
        // The row images a feed table's delete or update carries are for
        // the change feed, the rows are found by what names them
        ChangesetOp::Delete {
            table_id,
            columns,
            index_id,
            rows,
            ..
        } => apply_delete(ctx, *table_id, *columns, *index_id, rows).await,
        ChangesetOp::Update {
            table_id,
            columns,
            index_id,
            rows,
            ..
        } => apply_update(ctx, *table_id, *columns, *index_id, rows).await,
        other => Err(ZyronError::Internal(format!(
            "this operation is applied above the executor, not here: {other:?}"
        ))),
    }
}

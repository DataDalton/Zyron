//! Mirroring writes into a shadow table while a rewrite copies into it.
//!
//! An incompatible type change cannot be answered by reading the old bytes a
//! new way, so the rows are re-encoded into a second heap and the catalog is
//! pointed at it in one step. The copy takes time, and the table stays open the
//! whole time, so every write the source takes while the copy runs has to reach
//! the shadow too. That is what this does.
//!
//! The mirror runs off the same per-table maintenance list index maintenance
//! walks, so a writer that resolves its index set also resolves its shadow set,
//! and the wait an online DDL performs before it starts copying covers both the
//! same way.
//!
//! A value the shadow's column cannot hold aborts the writer's statement. The
//! alternative is a shadow that silently disagrees with the source, and the
//! swap would then install a table missing rows the writer was told it wrote.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use zyron_catalog::{ShadowSpec, TableEntry};
use zyron_common::page::PageId;
use zyron_common::{Result, RowLocator, TypeId, ZyronError};
use zyron_storage::TupleId;

use crate::batch::{DataBatch, batch_to_tuples};
use crate::context::ExecutionContext;

/// Packs a heap tuple id into the one word the row map keys and values on.
///
/// A heap row is a page number and a slot, and the page number never reaches
/// the top sixteen bits in a file the engine can address, so both fit one u64
/// and the map costs one word per row rather than an enum plus a struct.
#[inline]
fn pack(tuple: TupleId) -> u64 {
    (tuple.page_id.page_num << 16) | tuple.slot_id as u64
}

#[inline]
fn unpack(word: u64, file_id: u32) -> TupleId {
    TupleId::new(PageId::new(file_id, word >> 16), (word & 0xFFFF) as u16)
}

/// Packs a source row's address the same way, from the locator the index and
/// DML paths already carry.
#[inline]
pub fn pack_locator(locator: RowLocator) -> Option<u64> {
    match locator {
        RowLocator::Heap { page, slot } => Some((page.page_num << 16) | slot as u64),
        _ => None,
    }
}

/// Casts one column of a batch into the shape the shadow's copy holds.
///
/// A cast that cannot represent a value fails here, and the caller turns that
/// into the writer's own error rather than a rewrite that quietly drops the
/// row.
/// Only the one column comes back, so the batch the writer already holds is
/// not copied column by column to change one of them.
fn cast_for_shadow(
    batch: &DataBatch,
    position: usize,
    spec: &ShadowSpec,
) -> Result<crate::column::Column> {
    let source = &batch.columns[position];
    if spec.target_type == TypeId::Decimal {
        crate::compute::cast_column_to_decimal(source, spec.target_digits.unwrap_or(0))
    } else {
        crate::compute::cast_column(source, spec.target_type)
    }
}

/// Writes the rows of `batch` into every shadow the source table has, and
/// records where each one landed.
///
/// `source_ids` are the addresses the same rows took in the source heap, in the
/// same order, so a later update or delete on one of them can find its copy.
pub async fn mirror_insert(
    ctx: &Arc<ExecutionContext>,
    source: &TableEntry,
    shadows: &[ShadowSpec],
    batch: &DataBatch,
    source_ids: &[TupleId],
) -> Result<()> {
    for spec in shadows {
        if spec.abandoned.load(Ordering::Acquire) {
            continue;
        }
        let Some(position) = source.columns.iter().position(|c| c.id == spec.column_id) else {
            return Err(ZyronError::Internal(format!(
                "a shadow rewrite of table \"{}\" names column {} which the table does not have",
                source.name, spec.column_id.0
            )));
        };
        let shadow_entry = ctx.get_table_entry(spec.shadow_table_id)?;
        let column = cast_for_shadow(batch, position, spec).map_err(|e| {
            ZyronError::ExecutionError(format!(
                "value written to \"{}\".\"{}\" does not fit the type the running ALTER COLUMN \
                 SET TYPE is moving it to: {e}",
                source.name,
                source
                    .columns
                    .get(position)
                    .map(|c| c.name.as_str())
                    .unwrap_or("?")
            ))
        })?;
        let mut casted = batch.clone();
        casted.columns[position] = column;
        let tuples = batch_to_tuples(
            &casted,
            &shadow_entry.columns,
            ctx.txn_id,
            shadow_entry.schema_epoch,
        );
        let heap = ctx.get_heap_file(spec.shadow_table_id).await?;
        let written = heap.insert_batch(&tuples).await?;
        for (i, shadow_id) in written.iter().enumerate() {
            if let Some(source_id) = source_ids.get(i) {
                // Taken without an await apiece, so a writer mirroring a batch
                // does not suspend once per row against a map the copy is
                // filling at the same time
                let _ = spec.rows.insert_sync(pack(*source_id), pack(*shadow_id));
            }
            // The shadow's pages are dropped whole if this node restarts
            // before the swap, so they carry the WAL position the write
            // happened at rather than a record of their own
            ctx.buffer_pool
                .mark_dirty_with_lsn(shadow_id.page_id, ctx.wal.next_lsn().0);
        }
    }
    Ok(())
}

/// Stamps the shadow's copy of each deleted source row with the deleting
/// transaction, so the copy stops being visible at the same point the source's
/// does.
pub async fn mirror_delete(
    ctx: &Arc<ExecutionContext>,
    shadows: &[ShadowSpec],
    source_ids: &[TupleId],
) -> Result<()> {
    for spec in shadows {
        if spec.abandoned.load(Ordering::Acquire) {
            continue;
        }
        let shadow_entry = ctx.get_table_entry(spec.shadow_table_id)?;
        let heap = ctx.get_heap_file(spec.shadow_table_id).await?;
        for source_id in source_ids {
            let Some(word) = spec.rows.read_async(&pack(*source_id), |_, v| *v).await else {
                // The copy has not reached this row yet, so it will read the
                // source under the copy's snapshot and see the delete there
                continue;
            };
            let shadow_id = unpack(word, shadow_entry.heap_file_id);
            heap.set_xmax(shadow_id, ctx.txn_id).await?;
            // The shadow's pages are dropped whole if this node restarts
            // before the swap, so they carry the WAL position the write
            // happened at rather than a record of their own
            ctx.buffer_pool
                .mark_dirty_with_lsn(shadow_id.page_id, ctx.wal.next_lsn().0);
        }
    }
    Ok(())
}

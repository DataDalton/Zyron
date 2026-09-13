//! Branch-local copy-on-write write path.
//!
//! When a session has a branch active, DML routes here instead of mutating the
//! main heap. Inserts land in the branch's append heap file. Deletes and the
//! delete half of updates copy the target page into the branch's cow file once
//! and tombstone the slot in the copy, so the main line is never touched. Rows
//! that already live on a branch file are mutated in place.

use std::collections::HashMap;
use std::sync::Arc;

use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_common::{Result, ZyronError};
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{HeapPage, SlotId, Tuple, TupleId};

use crate::column::ScalarValue;
use crate::context::ExecutionContext;
use crate::operator::Operator;
use crate::operator::modify::InsertOperator;
use crate::operator::scan::{SeqScanOperator, read_page_through_pool};

/// Inserts tuples into the active branch's append heap file. Returns tuple ids
/// that carry the append file id, and publishes the new append page count so
/// scans pick up the rows.
pub(crate) async fn branch_insert(
    ctx: &ExecutionContext,
    branch_id: u64,
    heap_file_id: u32,
    tuples: &[Tuple],
) -> Result<Vec<TupleId>> {
    if tuples.is_empty() {
        return Ok(Vec::new());
    }
    let cat = ctx
        .branch_catalog
        .as_ref()
        .ok_or_else(|| ZyronError::Internal("branch insert requires a branch catalog".into()))?;
    let files = cat.branch_files_for(branch_id, heap_file_id);
    let heap = ctx
        .branch_append_heap(files.append_file_id, files.append_fsm_file_id)
        .await?;
    let ids = heap.insert_batch(tuples).await?;
    cat.set_append_page_count(branch_id, heap_file_id, heap.num_pages_cached() as u64);
    Ok(ids)
}

/// Deletes tuples under the active branch. Tuples on the main heap file are
/// copied into the branch cow file once and the slot is tombstoned in the copy.
/// Tuples already on a branch overlay file are tombstoned in place. Returns the
/// number of slots tombstoned.
pub(crate) async fn branch_delete(
    ctx: &ExecutionContext,
    branch_id: u64,
    heap_file_id: u32,
    tuple_ids: &[TupleId],
) -> Result<usize> {
    if tuple_ids.is_empty() {
        return Ok(0);
    }
    let cat = ctx
        .branch_catalog
        .as_ref()
        .ok_or_else(|| ZyronError::Internal("branch delete requires a branch catalog".into()))?;
    let files = cat.branch_files_for(branch_id, heap_file_id);

    // Group slots by their source page so each page is read and written once.
    let mut by_page: HashMap<PageId, Vec<u16>> = HashMap::new();
    for tid in tuple_ids {
        by_page.entry(tid.page_id).or_default().push(tid.slot_id);
    }

    let mut deleted = 0usize;
    for (page_id, slots) in by_page {
        // A main page must be copied into the branch first; a page already on a
        // branch overlay (cow copy or append) is mutated directly.
        let target = if page_id.file_id == heap_file_id {
            cow_copy_page(ctx, cat.as_ref(), branch_id, files.cow_file_id, page_id).await?
        } else {
            page_id
        };

        let data = read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, target).await?;
        let mut page = HeapPage::from_bytes(data);
        let mut modified = false;
        for slot in slots {
            if page.delete_tuple(SlotId(slot)) {
                deleted += 1;
                modified = true;
            }
        }
        if modified {
            write_page_through_pool(ctx, target, page.as_bytes()).await?;
        }
    }
    Ok(deleted)
}

/// Ensures the branch has a copy-on-write copy of `original_page` in its cow
/// file, returning the branch-local page id. Idempotent: a second caller for the
/// same page reuses the first copy.
async fn cow_copy_page(
    ctx: &ExecutionContext,
    cat: &dyn zyron_common::BranchCatalog,
    branch_id: u64,
    cow_file_id: u32,
    original_page: PageId,
) -> Result<PageId> {
    if let Some(existing) = cat.lookup_cow_page(branch_id, original_page) {
        return Ok(existing);
    }
    // Copy from the branch's currently visible version of the page: a parent's
    // cow copy when present, otherwise the main page.
    let source = cat.resolve_page_for(branch_id, original_page);
    let bytes = read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, source).await?;
    let local = ctx.disk_manager.allocate_page(cow_file_id).await?;
    write_page_through_pool(ctx, local, &bytes).await?;
    // Another writer may have copied the same page concurrently; the recorded
    // winner is authoritative and our freshly written page leaks harmlessly.
    Ok(cat.record_cow_page(branch_id, original_page, local))
}

/// Result of merging one table's branch overlay into the main line.
#[derive(Debug, Default, Clone, Copy)]
pub struct BranchMergeStats {
    pub inserted: u64,
    pub deleted: u64,
}

/// Builds logical columns for every table column in table order, used to scan a
/// branch's appended rows with the full row image for re-insert onto main.
fn all_table_columns(table_entry: &zyron_catalog::TableEntry) -> Vec<LogicalColumn> {
    table_entry
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

/// Materializes one table's branch overlay onto the main line. Branch row
/// tombstones (cow pages with slots emptied versus the main page) are applied to
/// the main heap, and branch-inserted rows are replayed through the standard
/// insert path so every main index is maintained. The context must NOT have an
/// active branch so the writes land on the main line.
///
/// Stale main index entries left pointing at tombstoned rows are harmless: the
/// index scan rechecks visibility and deletion when it fetches each tuple.
pub async fn merge_branch_table_into_main(
    ctx: &Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    cow_overrides: &[(PageId, PageId)],
    append_file_id: u32,
    append_page_count: u64,
) -> Result<BranchMergeStats> {
    let table_entry = ctx.get_table_entry(table_id)?;
    let main_heap = ctx.get_heap_file(table_id).await?;
    let txn_id = ctx.txn_id;
    let mut stats = BranchMergeStats::default();

    // Apply branch row tombstones: a slot live on the main page but emptied on
    // the branch cow copy was deleted (or updated, which deletes then appends)
    // by the branch.
    let mut del_tids: Vec<TupleId> = Vec::new();
    // The rows the merge removes from the main line, as the main pages hold
    // them, for the table's change feed
    let mut del_images: Vec<Vec<u8>> = Vec::new();
    for (original, cow) in cow_overrides {
        let main_data =
            read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, *original).await?;
        let cow_data = read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, *cow).await?;
        let main_hdr = HeapPage::heap_header_from_slice(&main_data);
        let cow_hdr = HeapPage::heap_header_from_slice(&cow_data);
        let main_page = HeapPage::from_bytes(main_data);
        let cow_page = HeapPage::from_bytes(cow_data);
        let limit = main_hdr.slot_count.min(cow_hdr.slot_count);
        for s in 0..limit {
            let slot = SlotId(s);
            let main_live = main_page
                .get_slot(slot)
                .map(|x| !x.is_empty())
                .unwrap_or(false);
            let cow_live = cow_page
                .get_slot(slot)
                .map(|x| !x.is_empty())
                .unwrap_or(false);
            if main_live && !cow_live {
                del_tids.push(TupleId::new(*original, s));
                if let Some(view) = main_page.get_tuple_view(slot) {
                    del_images.push(view.data.to_vec());
                }
            }
        }
    }
    if !del_tids.is_empty() {
        // MVCC delete on the main line: stamp xmax with the merge transaction.
        // Index entries are kept; the index scan rechecks visibility and key on
        // fetch, so the merged-out rows drop out without entry removal. The
        // heap logs each page it stamps and stamps the page with the record
        stats.deleted = main_heap
            .mark_deleted_batch(
                &del_tids,
                txn_id,
                ctx.snapshot.prune_horizon(),
                Some(ctx.snapshot.status_map().as_ref()),
                // The retention-aware background vacuum is the authoritative
                // enforcer for the table; the just-stamped merge rows sit above
                // the prune horizon, so on-access pruning does not touch them.
                false,
            )
            .await? as u64;
        ctx.mark_wrote_wal();
        // A merge changes the main line the way a delete of these rows
        // would, so the table's feed records it as one
        if let Some(capture) = ctx.change_capture(ctx.wal.next_lsn().0) {
            let refs: Vec<&[u8]> = del_images.iter().map(|v| v.as_slice()).collect();
            capture
                .hook
                .on_delete(
                    table_id.0,
                    &refs,
                    capture.version,
                    capture.timestamp,
                    txn_id,
                    true,
                    capture.branch,
                )
                .map_err(|e| ZyronError::ExecutionError(format!("CDC delete hook failed: {e}")))?;
        }
    }

    // Replay branch-inserted rows onto main through the standard insert path so
    // every index is maintained. The source scans only the branch append file.
    if append_page_count > 0 {
        let columns = all_table_columns(&table_entry);
        let source = SeqScanOperator::new(ctx.clone(), table_id, columns, None, false, None)
            .await?
            .scan_append_file(append_file_id, append_page_count);
        // The scan produces every table column in order, so the insert maps
        // one-to-one (the identity reshape fast path).
        let full_targets: Vec<zyron_catalog::ColumnId> =
            table_entry.columns.iter().map(|c| c.id).collect();
        // Re-enforce CHECK constraints on merge: a constraint added after the
        // branch rows were inserted must still hold on the main line. Defaults
        // do not re-apply (every column is supplied by the scan).
        let checks =
            zyron_planner::bind_table_check_constraints(&ctx.catalog, &table_entry).await?;
        let mut insert_op = InsertOperator::new(
            Box::new(source),
            ctx.clone(),
            table_id,
            full_targets,
            Vec::new(),
            checks,
            Vec::new(),
            // Stored generated values were computed when the rows entered
            // the branch, the replay carries them as data
            Vec::new(),
        );
        if let Some(batch) = insert_op.next().await? {
            if let Some(col) = batch.batch.columns.first() {
                if let ScalarValue::Int64(v) = col.data.get_scalar(0) {
                    stats.inserted = v.max(0) as u64;
                }
            }
        }
    }

    Ok(stats)
}

/// Writes a page through the buffer pool and marks it dirty so the background
/// writer flushes it to the branch file.
async fn write_page_through_pool(
    ctx: &ExecutionContext,
    page_id: PageId,
    data: &[u8; PAGE_SIZE],
) -> Result<()> {
    if let Some(frame) = ctx.buffer_pool.fetch_page(page_id) {
        frame.copy_from(data);
        ctx.buffer_pool.unpin_page(page_id, true);
        return Ok(());
    }
    ctx.buffer_pool.load_page(page_id, data)?;
    ctx.buffer_pool.unpin_page(page_id, true);
    Ok(())
}

//! DML operators for INSERT, UPDATE, and DELETE.
//!
//! Each operator pulls rows from a child, performs the storage mutation
//! (with WAL logging), and returns a single-row batch containing the
//! affected row count.

use std::sync::Arc;

use zyron_common::{TypeId, ZyronError};
use zyron_planner::binder::{BoundAssignment, BoundExpr};
use zyron_planner::logical::LogicalColumn;
use zyron_storage::TupleId;

use crate::batch::{
    DataBatch, batch_to_tuples, build_column_to_builder_map, create_builders,
    decode_tuple_into_builders, encode_scalar_value, finalize_builders,
};
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use zyron_storage::columnar::{ColumnarPatchManager, PatchStore};

/// Returns the per-table columnar patch store. Used by UPDATE and DELETE to
/// route mutations of columnar-resident rows to the append-only patch log
/// instead of the heap, with no .zyr rewrite and no heap round trip. Also
/// used by the row-locking operator's post-wait staleness recheck.
pub(crate) fn columnar_patch_store(
    te: &zyron_catalog::TableEntry,
) -> zyron_common::Result<Arc<PatchStore>> {
    let seg = te.columnar.segments.first().ok_or_else(|| {
        ZyronError::Internal("columnar locators present but no registered segments".into())
    })?;
    ColumnarPatchManager::store_for_segment(te.id.0 as u64, std::path::Path::new(&seg.path))
}
use crate::context::ExecutionContext;
use crate::expr::{evaluate, literal_to_scalar};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Encodes a row's value at the given column position into a caller-provided
/// buffer suitable for B+Tree key comparison. Big-endian for integers (matches
/// the literal path in extract_scan_bounds), Utf8 as raw bytes
///
/// Returns true when the row contributed a key, false when the column is
/// missing or null. Caller is expected to hoist buf out of any row loop and
/// clear it inside this function so a single Vec amortizes across the batch
pub(crate) fn encode_btree_key_into(
    batch: &DataBatch,
    row_idx: usize,
    col_pos: usize,
    type_id: TypeId,
    buf: &mut Vec<u8>,
) -> bool {
    let Some(col) = batch.columns.get(col_pos) else {
        return false;
    };
    if col.is_null(row_idx) {
        return false;
    }
    buf.clear();
    match (&col.data, type_id) {
        (ColumnData::Int8(v), _) => {
            buf.extend_from_slice(&(v[row_idx] as i64 as u64).to_be_bytes())
        }
        (ColumnData::Int16(v), _) => {
            buf.extend_from_slice(&(v[row_idx] as i64 as u64).to_be_bytes())
        }
        (ColumnData::Int32(v), _) => {
            buf.extend_from_slice(&(v[row_idx] as i64 as u64).to_be_bytes())
        }
        (ColumnData::Int64(v), _) => buf.extend_from_slice(&(v[row_idx] as u64).to_be_bytes()),
        // 16-byte order-preserving key for Int128 (incl. i128 picosecond
        // timestamps). The explicit sign-bit flip keeps negative values
        // (pre-1970 timestamps) ordered before positive ones, which the bare
        // cast used for the narrower widths only achieves for non-negatives
        (ColumnData::Int128(v), _) => {
            let key = (v[row_idx] as u128) ^ (1u128 << 127);
            buf.extend_from_slice(&key.to_be_bytes());
        }
        (ColumnData::UInt8(v), _) => buf.extend_from_slice(&(v[row_idx] as u64).to_be_bytes()),
        (ColumnData::UInt16(v), _) => buf.extend_from_slice(&(v[row_idx] as u64).to_be_bytes()),
        (ColumnData::UInt32(v), _) => buf.extend_from_slice(&(v[row_idx] as u64).to_be_bytes()),
        (ColumnData::UInt64(v), _) => buf.extend_from_slice(&v[row_idx].to_be_bytes()),
        (ColumnData::Float64(v), _) => {
            let bits = v[row_idx].to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            buf.extend_from_slice(&sortable.to_be_bytes());
        }
        (ColumnData::Float32(v), _) => {
            let bits = (v[row_idx] as f64).to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            buf.extend_from_slice(&sortable.to_be_bytes());
        }
        (ColumnData::Utf8(v), _) => buf.extend_from_slice(v[row_idx].as_bytes()),
        (ColumnData::Binary(v), _) => buf.extend_from_slice(&v[row_idx]),
        _ => return false,
    }
    true
}

/// Length of the locator suffix appended to every B+tree index key. Distinct
/// rows that share an indexed value get distinct, ordered composite keys
/// (value bytes followed by the row locator's order-preserving encoding), so
/// the unique-key B+tree stores a non-unique secondary index without
/// collisions. The encoding is fixed width, so range bounds pad with all-0x00
/// and all-0xFF suffixes and exact-value checks work from the length alone
pub(crate) const INDEX_LOCATOR_SUFFIX_LEN: usize = zyron_common::RowLocator::ENCODED_LEN;

/// The btree-normalized locator of a heap row: the heap file_id is implicit
/// from the index, so keys and payloads carry page number and slot only
#[inline]
fn btree_heap_locator(page_num: u64, slot: u16) -> zyron_common::RowLocator {
    zyron_common::RowLocator::Heap {
        page: zyron_common::page::PageId::new(0, page_num),
        slot,
    }
}

/// Normalizes any locator for btree storage, stripping the heap file_id
#[inline]
pub(crate) fn btree_normalize_locator(loc: zyron_common::RowLocator) -> zyron_common::RowLocator {
    match loc {
        zyron_common::RowLocator::Heap { page, slot } => btree_heap_locator(page.page_num, slot),
        other => other,
    }
}

/// Encodes a raw storage cell (little-endian fixed value or bare varlen
/// bytes, the NSM and .zyr cell form) into the same order-preserving key
/// bytes `encode_btree_key_into` produces for the equivalent batch column.
/// Dispatches on the column's physical type, which is what the cell layout
/// follows. The parity is enforced by a unit test, a drifted encoding would
/// make key rebuilds silently miss their entries
pub fn encode_btree_key_from_cell(phys: TypeId, cell: &[u8], buf: &mut Vec<u8>) -> bool {
    buf.clear();
    let sortable_f64 = |f: f64, buf: &mut Vec<u8>| {
        let bits = f.to_bits();
        let sortable = if bits >> 63 == 1 {
            !bits
        } else {
            bits ^ (1u64 << 63)
        };
        buf.extend_from_slice(&sortable.to_be_bytes());
    };
    match phys {
        TypeId::Int8 if cell.len() == 1 => {
            buf.extend_from_slice(&(cell[0] as i8 as i64 as u64).to_be_bytes());
        }
        TypeId::Int16 if cell.len() == 2 => {
            let v = i16::from_le_bytes([cell[0], cell[1]]);
            buf.extend_from_slice(&(v as i64 as u64).to_be_bytes());
        }
        TypeId::Int32 if cell.len() == 4 => {
            let v = i32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
            buf.extend_from_slice(&(v as i64 as u64).to_be_bytes());
        }
        TypeId::Int64 if cell.len() == 8 => {
            let Ok(b) = <[u8; 8]>::try_from(cell) else {
                return false;
            };
            buf.extend_from_slice(&(i64::from_le_bytes(b) as u64).to_be_bytes());
        }
        TypeId::Int128 if cell.len() == 16 => {
            let Ok(b) = <[u8; 16]>::try_from(cell) else {
                return false;
            };
            let key = (i128::from_le_bytes(b) as u128) ^ (1u128 << 127);
            buf.extend_from_slice(&key.to_be_bytes());
        }
        TypeId::UInt8 if cell.len() == 1 => {
            buf.extend_from_slice(&(cell[0] as u64).to_be_bytes());
        }
        TypeId::UInt16 if cell.len() == 2 => {
            let v = u16::from_le_bytes([cell[0], cell[1]]);
            buf.extend_from_slice(&(v as u64).to_be_bytes());
        }
        TypeId::UInt32 if cell.len() == 4 => {
            let v = u32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
            buf.extend_from_slice(&(v as u64).to_be_bytes());
        }
        TypeId::UInt64 if cell.len() == 8 => {
            let Ok(b) = <[u8; 8]>::try_from(cell) else {
                return false;
            };
            buf.extend_from_slice(&u64::from_le_bytes(b).to_be_bytes());
        }
        TypeId::Float64 if cell.len() == 8 => {
            let Ok(b) = <[u8; 8]>::try_from(cell) else {
                return false;
            };
            sortable_f64(f64::from_le_bytes(b), buf);
        }
        TypeId::Float32 if cell.len() == 4 => {
            let v = f32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]]);
            sortable_f64(v as f64, buf);
        }
        _ if phys.fixed_size().is_none() => {
            // varlen cells carry the raw bytes, exactly the Utf8/Binary key form
            buf.extend_from_slice(cell);
        }
        _ => return false,
    }
    true
}

/// Re-keys every B+tree entry for rows a fold moved from the heap into a
/// columnar segment: the heap-suffixed key is deleted and the columnar
/// suffixed twin inserted, so index scans keep serving the rows after their
/// heap slots are zeroed. `indexed_cells` carries each indexed column's raw
/// cell bytes per folded row, aligned with `folded`, whose position i holds
/// sys_rowid `base_rowid + i` by construction
pub fn fold_rekey_btree_entries(
    folded: &[(zyron_common::page::PageId, u16, u32)],
    indexed_cells: &[(zyron_catalog::ColumnId, TypeId, Vec<Option<&[u8]>>)],
    file_id: u64,
    base_rowid: u64,
    btree: &[(zyron_catalog::IndexId, zyron_catalog::ColumnId, bool)],
    registry: &scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>,
) {
    if folded.is_empty() || btree.is_empty() {
        return;
    }
    let mut scratch: Vec<u8> = Vec::with_capacity(48);
    for (idx_id, col_id, _unique) in btree {
        let Some(tree) = registry.read_sync(&idx_id.0, |_, v| Arc::clone(v)) else {
            continue;
        };
        let Some((_, phys, cells)) = indexed_cells.iter().find(|(cid, _, _)| cid == col_id) else {
            continue;
        };
        for (i, (page_id, slot, _xmin)) in folded.iter().enumerate() {
            let Some(cell) = cells[i] else {
                continue;
            };
            if !encode_btree_key_from_cell(*phys, cell, &mut scratch) {
                continue;
            }
            let value_len = scratch.len();
            btree_heap_locator(page_id.page_num, *slot).append_key_suffix(&mut scratch);
            tree.delete_sync(&scratch);
            scratch.truncate(value_len);
            let loc = zyron_common::RowLocator::Columnar {
                file_id,
                sys_rowid: base_rowid + i as u64,
            };
            loc.append_key_suffix(&mut scratch);
            if let Err(e) = tree.insert_sync(&scratch, loc) {
                // a duplicate is a benign redo, anything else is data loss
                if !matches!(e, zyron_common::ZyronError::DuplicateKey) {
                    eprintln!("fold btree re-key insert failed on index {}: {e}", idx_id.0);
                }
            }
        }
    }
}

/// True when the columnar row's current visible value for the indexed column
/// still encodes to `value_key`. Old-value index entries survive an update
/// (the locator does not change), so they must never prove a conflict
async fn columnar_row_currently_holds_value(
    ctx: &Arc<ExecutionContext>,
    te: &zyron_catalog::TableEntry,
    col_pos: usize,
    file_id: u64,
    sys_rowid: u64,
    value_key: &[u8],
) -> zyron_common::Result<bool> {
    let c = &te.columns[col_pos];
    let logical = vec![LogicalColumn {
        table_idx: Some(0),
        column_id: c.id,
        name: c.name.clone(),
        type_id: c.type_id,
        nullable: c.nullable,
        ts_precision: c.ts_precision,
    }];
    let loc = [zyron_common::RowLocator::Columnar { file_id, sys_rowid }];
    let fetcher = crate::operator::doc_fetch::DocRowFetcher::prepare_columnar_only(
        ctx, te.id, &logical, &loc, None,
    )
    .await?;
    let Some(vals) = fetcher.columnar_row(file_id, sys_rowid) else {
        return Ok(false);
    };
    let mut builders = create_builders(&logical, 1);
    builders[0].push(&vals[0]);
    let row = finalize_builders(builders);
    let mut buf = Vec::with_capacity(24);
    Ok(encode_btree_key_into(&row, 0, 0, c.type_id, &mut buf) && buf == value_key)
}

/// Latest-committed liveness of a columnar row for constraint probes: the
/// folded base version is committed by construction, so the row is dead only
/// when a committed supersede exists on the branch this statement writes
pub(crate) fn columnar_row_live_latest(
    ctx: &ExecutionContext,
    te: &zyron_catalog::TableEntry,
    file_id: u64,
    sys_rowid: u64,
) -> zyron_common::Result<bool> {
    let store = columnar_patch_store(te)?;
    let overlay = match ctx.active_branch_id {
        Some(branch) => store.row_overlay_on(branch, file_id, sys_rowid),
        None => store.row_overlay(file_id, sys_rowid),
    };
    let Some(overlay) = overlay else {
        return Ok(true);
    };
    let status = ctx.snapshot.status_map();
    Ok(!overlay.supersedes.iter().any(|&x| status.is_committed(x)))
}

/// Builds a unique-constraint violation error for a table column.
fn unique_violation(
    table_entry: &zyron_catalog::TableEntry,
    col_pos: usize,
) -> zyron_common::ZyronError {
    zyron_common::ZyronError::UniqueViolation(format!(
        "duplicate key value violates unique constraint on \"{}\".\"{}\"",
        table_entry.name, table_entry.columns[col_pos].name
    ))
}

/// Enforces unique B+tree constraints for a batch of rows BEFORE any heap or
/// index mutation, so a violation aborts the statement with no partial writes
/// (transaction abort does not undo heap rows here). A value conflicts when a
/// LIVE committed row (not in `exclude_tids`, which are the rows an UPDATE is
/// replacing) already holds it, or when it appears twice within this batch.
/// Index entries of MVCC-deleted rows are skipped by fetching each candidate and
/// testing latest-committed liveness. Null values are unconstrained.
pub(crate) async fn check_unique_constraints(
    ctx: &Arc<ExecutionContext>,
    table_entry: &zyron_catalog::TableEntry,
    batch: &DataBatch,
    index_snap: &zyron_catalog::TableIndexSnapshot,
    exclude_locators: &[zyron_common::RowLocator],
) -> zyron_common::Result<()> {
    if index_snap.btree.is_empty() {
        return Ok(());
    }
    let has_unique = index_snap.btree.iter().any(|(_, _, u)| *u);
    if !has_unique {
        return Ok(());
    }
    // Stored entries carry btree-normalized locators, so the exclusion set
    // must too or an updated row would conflict with its own old entry
    let exclude: std::collections::HashSet<zyron_common::RowLocator> = exclude_locators
        .iter()
        .map(|l| btree_normalize_locator(*l))
        .collect();
    let heap_file_id = table_entry.heap_file_id;
    let mut scratch: Vec<u8> = Vec::with_capacity(24);
    for (idx_id, col_id, unique) in &index_snap.btree {
        if !*unique {
            continue;
        }
        let Some(btree) = ctx.get_index(*idx_id) else {
            continue;
        };
        let Some(col_pos) = table_entry.columns.iter().position(|c| c.id == *col_id) else {
            continue;
        };
        let col_type = table_entry.columns[col_pos].type_id;
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for row_idx in 0..batch.num_rows {
            scratch.clear();
            if !encode_btree_key_into(batch, row_idx, col_pos, col_type, &mut scratch) {
                continue; // null is unconstrained
            }
            // Serialize concurrent inserters of the same value: take a key lock
            // (namespaced by index id) held until commit/abort. A concurrent
            // transaction holding it gets a conflict, so two committed rows with
            // the same unique value can never both win the race.
            if let Some(locks) = &ctx.intent_locks {
                let mut lock_key = Vec::with_capacity(4 + scratch.len());
                lock_key.extend_from_slice(&idx_id.0.to_le_bytes());
                lock_key.extend_from_slice(&scratch);
                locks.lock_key(ctx.txn_id as u64, table_entry.id.0, &lock_key)?;
            }
            if !seen.insert(scratch.clone()) {
                return Err(unique_violation(table_entry, col_pos));
            }
            // Collect index candidates for this value, excluding the rows being
            // updated, then confirm each points at a LIVE committed row (an
            // MVCC-deleted row's stale entry is not a conflict).
            let mut lo = scratch.clone();
            lo.extend_from_slice(&[0u8; INDEX_LOCATOR_SUFFIX_LEN]);
            let mut hi = scratch.clone();
            hi.extend_from_slice(&[0xFFu8; INDEX_LOCATOR_SUFFIX_LEN]);
            let value_len = scratch.len();
            let mut candidates: Vec<zyron_common::RowLocator> = Vec::new();
            btree.range_scan_for_each(Some(&lo), Some(&hi), |k, existing| {
                // A composite key is value||locator_suffix. For a variable
                // length value, the range can also catch a longer value sharing
                // this prefix, so require an exact value match (length + bytes)
                let exact_value = k.len() == value_len + INDEX_LOCATOR_SUFFIX_LEN
                    && k[..value_len] == scratch[..];
                if exact_value && !exclude.contains(&existing) {
                    candidates.push(existing);
                }
                true
            });
            for cand in candidates {
                let live = match cand {
                    zyron_common::RowLocator::Heap { page, slot } => {
                        let page_id = zyron_common::page::PageId::new(heap_file_id, page.page_num);
                        let data = crate::operator::scan::read_page_through_pool(
                            &ctx.buffer_pool,
                            &ctx.disk_manager,
                            page_id,
                        )
                        .await?;
                        match zyron_storage::HeapPage::get_tuple_view_from_slice(
                            &data,
                            zyron_storage::SlotId(slot),
                        ) {
                            Some(view) => ctx
                                .snapshot
                                .is_live_latest(view.header.xmin as u64, view.header.xmax as u64),
                            None => false,
                        }
                    }
                    zyron_common::RowLocator::Columnar { file_id, sys_rowid } => {
                        // an updated columnar row keeps its locator, so a
                        // live candidate found under a stale value must be
                        // refuted by the row's current visible value
                        columnar_row_live_latest(ctx, table_entry, file_id, sys_rowid)?
                            && columnar_row_currently_holds_value(
                                ctx,
                                table_entry,
                                col_pos,
                                file_id,
                                sys_rowid,
                                &scratch[..value_len],
                            )
                            .await?
                    }
                    zyron_common::RowLocator::Lake { .. } => false,
                };
                if live {
                    return Err(unique_violation(table_entry, col_pos));
                }
            }
        }
    }
    Ok(())
}

/// Adds B+tree index entries for a batch of newly stored rows. Each entry's key
/// is the indexed value followed by the row's locator suffix, so two rows with
/// the same value coexist (non-unique index support); composite keys are globally
/// unique by locator. Unique constraints are enforced separately by
/// `check_unique_constraints` before the rows are written.
///
/// `batch` must have its columns in table-column order (the INSERT source and the
/// UPDATE/DELETE scans both project all columns in that order), so an index
/// column is found at its position within `table_entry.columns`.
pub(crate) fn maintain_btree_insert(
    ctx: &ExecutionContext,
    table_entry: &zyron_catalog::TableEntry,
    batch: &DataBatch,
    locators: &[zyron_common::RowLocator],
    index_snap: &zyron_catalog::TableIndexSnapshot,
) {
    if index_snap.btree.is_empty() {
        return;
    }
    let mut key_bytes: Vec<u8> = Vec::with_capacity(locators.len() * 32);
    let mut key_spans: Vec<(usize, usize, zyron_common::RowLocator)> =
        Vec::with_capacity(locators.len());
    let mut scratch: Vec<u8> = Vec::with_capacity(48);
    for (idx_id, col_id, _unique) in &index_snap.btree {
        let Some(btree) = ctx.get_index(*idx_id) else {
            continue;
        };
        let Some(col_pos) = table_entry.columns.iter().position(|c| c.id == *col_id) else {
            continue;
        };
        let col_type = table_entry.columns[col_pos].type_id;
        key_bytes.clear();
        key_spans.clear();
        for (row_idx, loc) in locators.iter().enumerate() {
            scratch.clear();
            if encode_btree_key_into(batch, row_idx, col_pos, col_type, &mut scratch) {
                let normalized = btree_normalize_locator(*loc);
                normalized.append_key_suffix(&mut scratch);
                let off = key_bytes.len();
                key_bytes.extend_from_slice(&scratch);
                key_spans.push((off, scratch.len(), normalized));
            }
        }
        if !key_spans.is_empty() {
            let mut items: Vec<(&[u8], zyron_common::RowLocator)> = key_spans
                .iter()
                .map(|&(off, len, loc)| (&key_bytes[off..off + len], loc))
                .collect();
            // Composite keys are globally unique, so the batch never collides for
            // distinct rows; the per-key fallback covers a benign re-insert.
            if btree.insert_many(&mut items).is_err() {
                for &(off, len, loc) in &key_spans {
                    let _ = btree.insert_sync(&key_bytes[off..off + len], loc);
                }
            }
        }
    }
}

/// Computes a table's effective retention floor: the lowest commit LSN that
/// must remain reconstructable, so vacuum reclaims a committed-delete tuple only
/// when its deleter committed at or below it. Combines two sources, keeping the
/// older (lower) of the two so either keeps a version alive:
///   - version tags (CREATE VERSION), an instance-wide floor on the CLOG,
///   - this table's time-travel retention window, mapped to an LSN via the
///     retention clock: 0 secs means no time retention (u64::MAX, time never
///     keeps anything), u64::MAX means unlimited (floor 0, keep everything),
///     and a finite window resolves to the LSN as of `now - window`.
pub fn effective_retention_floor(
    table: &zyron_catalog::TableEntry,
    status_map: &zyron_storage::TxnStatusMap,
    clock: &zyron_storage::RetentionClock,
    now_micros: u64,
) -> u64 {
    let tag_floor = status_map.version_retention_floor();
    let time_floor = match table.time_travel_retention_secs {
        0 => u64::MAX,
        u64::MAX => 0,
        secs => {
            let cutoff = now_micros.saturating_sub(secs.saturating_mul(1_000_000));
            clock.lsn_at(cutoff)
        }
    };
    tag_floor.min(time_floor)
}

/// Removes B+tree index entries for rows that vacuum reclaimed from the heap.
///
/// On an MVCC delete or update the old row's index entries are intentionally
/// kept: a still-live snapshot can read the old version, and an index scan
/// rechecks visibility and the composite key on fetch. Once vacuum reclaims the
/// dead heap tuple no snapshot can reach it, so its entries are pure bloat and
/// are removed here. `dead` carries each reclaimed tuple's (slot, row image)
/// captured before the heap slot was zeroed, so the composite key (indexed
/// value followed by the row's tuple id) is rebuilt from the same value that
/// was indexed and deleted exactly. Entries for live rows that share an indexed
/// value keep their own distinct tuple-id suffix and are untouched.
pub fn vacuum_index_cleanup(
    table_entry: &zyron_catalog::TableEntry,
    page_id: zyron_common::page::PageId,
    dead: &[(u16, Vec<u8>)],
    btree: &[(zyron_catalog::IndexId, zyron_catalog::ColumnId, bool)],
    registry: &scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>,
) {
    if dead.is_empty() || btree.is_empty() {
        return;
    }
    // Decode the reclaimed rows into a batch with every table column in order,
    // so a column position indexes both the table and the batch.
    let logical: Vec<LogicalColumn> = table_entry
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            ts_precision: c.ts_precision,
        })
        .collect();
    let col_ids: Vec<zyron_catalog::ColumnId> = table_entry.columns.iter().map(|c| c.id).collect();
    let column_to_builder = build_column_to_builder_map(&table_entry.columns, &col_ids);
    let mut builders = create_builders(&logical, dead.len());
    for (_slot, row_data) in dead {
        decode_tuple_into_builders(
            row_data,
            &table_entry.columns,
            &column_to_builder,
            &mut builders,
        );
    }
    let batch = finalize_builders(builders);

    let mut scratch: Vec<u8> = Vec::with_capacity(24);
    for (idx_id, col_id, _unique) in btree {
        let Some(tree) = registry.read_sync(&idx_id.0, |_, v| Arc::clone(v)) else {
            continue;
        };
        let Some(col_pos) = table_entry.columns.iter().position(|c| c.id == *col_id) else {
            continue;
        };
        let col_type = table_entry.columns[col_pos].type_id;
        for (row_idx, (slot, _)) in dead.iter().enumerate() {
            scratch.clear();
            if encode_btree_key_into(&batch, row_idx, col_pos, col_type, &mut scratch) {
                btree_heap_locator(page_id.page_num, *slot).append_key_suffix(&mut scratch);
                tree.delete_sync(&scratch);
            }
        }
    }
}

/// Rebuilds one B+tree index's entries from a page's live rows during REINDEX.
/// `live` carries each live tuple's (slot, row image) read from the heap, in the
/// same table-column order the insert path produces. The composite key (the
/// indexed value followed by the row's tuple id) is encoded exactly as
/// `maintain_btree_insert` builds it, so the rebuilt tree is identical to one
/// populated incrementally. Returns the number of entries inserted.
pub fn rebuild_btree_index_from_rows(
    table_entry: &zyron_catalog::TableEntry,
    page_id: zyron_common::page::PageId,
    live: &[(u16, Vec<u8>)],
    col_id: zyron_catalog::ColumnId,
    btree: &Arc<zyron_storage::BTreeIndex>,
) -> usize {
    if live.is_empty() {
        return 0;
    }
    let Some(col_pos) = table_entry.columns.iter().position(|c| c.id == col_id) else {
        return 0;
    };
    let col_type = table_entry.columns[col_pos].type_id;

    // Decode the live rows into a batch with every table column in order, so a
    // column position indexes both the table and the batch.
    let logical: Vec<LogicalColumn> = table_entry
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            ts_precision: c.ts_precision,
        })
        .collect();
    let col_ids: Vec<zyron_catalog::ColumnId> = table_entry.columns.iter().map(|c| c.id).collect();
    let column_to_builder = build_column_to_builder_map(&table_entry.columns, &col_ids);
    let mut builders = create_builders(&logical, live.len());
    for (_slot, row_data) in live {
        decode_tuple_into_builders(
            row_data,
            &table_entry.columns,
            &column_to_builder,
            &mut builders,
        );
    }
    let batch = finalize_builders(builders);

    let mut inserted = 0usize;
    let mut scratch: Vec<u8> = Vec::with_capacity(24);
    for (row_idx, (slot, _)) in live.iter().enumerate() {
        scratch.clear();
        if encode_btree_key_into(&batch, row_idx, col_pos, col_type, &mut scratch) {
            let loc = btree_heap_locator(page_id.page_num, *slot);
            loc.append_key_suffix(&mut scratch);
            if btree.insert_sync(&scratch, loc).is_ok() {
                inserted += 1;
            }
        }
    }
    inserted
}

/// Rebuilds one B+tree index's entries from an already-materialized batch of
/// rows during REINDEX, used for columnar-resident rows whose values come
/// from the columnar scan. Batch columns are in table-column order and
/// `locators` is row-aligned. Returns the number of entries inserted.
pub fn rebuild_btree_index_from_batch(
    table_entry: &zyron_catalog::TableEntry,
    batch: &DataBatch,
    locators: &[zyron_common::RowLocator],
    col_id: zyron_catalog::ColumnId,
    btree: &Arc<zyron_storage::BTreeIndex>,
) -> usize {
    let Some(col_pos) = table_entry.columns.iter().position(|c| c.id == col_id) else {
        return 0;
    };
    let col_type = table_entry.columns[col_pos].type_id;
    let mut inserted = 0usize;
    let mut scratch: Vec<u8> = Vec::with_capacity(48);
    for (row_idx, loc) in locators.iter().enumerate() {
        scratch.clear();
        if encode_btree_key_into(batch, row_idx, col_pos, col_type, &mut scratch) {
            let normalized = btree_normalize_locator(*loc);
            normalized.append_key_suffix(&mut scratch);
            if btree.insert_sync(&scratch, normalized).is_ok() {
                inserted += 1;
            }
        }
    }
    inserted
}

/// Extracts text content from a DataBatch row for FTS indexing into a reusable buffer.
/// Concatenates all text-type columns (Varchar, Text, Char) for the given row
/// into the buffer, separated by spaces. The caller should call buf.clear() between rows.
fn extract_fts_text_into(
    batch: &DataBatch,
    row_idx: usize,
    columns: &[zyron_catalog::ColumnEntry],
    buf: &mut String,
) {
    for (col_idx, col_entry) in columns.iter().enumerate() {
        match col_entry.type_id {
            TypeId::Varchar | TypeId::Text | TypeId::Char => {}
            _ => continue,
        }
        if col_idx >= batch.columns.len() {
            continue;
        }
        let col = &batch.columns[col_idx];
        if row_idx >= col.data.len() {
            continue;
        }
        if col.nulls.is_null(row_idx) {
            continue;
        }
        if let ColumnData::Utf8(ref strings) = col.data {
            if row_idx < strings.len() {
                if !buf.is_empty() {
                    buf.push(' ');
                }
                buf.push_str(&strings[row_idx]);
            }
        }
    }
}

/// Extracts the raw vector bytes for a specific column from a DataBatch row.
/// The target column is identified by its catalog ColumnId rather than by
/// position, so tables with multiple vector columns correctly route each
/// index's maintenance to its own column. Returns None if the column is not
/// present in the batch, is null, or is not a Vector column.
fn extract_vector_bytes<'a>(
    batch: &'a DataBatch,
    row_idx: usize,
    columns: &[zyron_catalog::ColumnEntry],
    target_column_id: u16,
) -> Option<&'a [u8]> {
    let col_idx = columns
        .iter()
        .position(|c| c.id.0 == target_column_id && c.type_id == TypeId::Vector)?;
    if col_idx >= batch.columns.len() {
        return None;
    }
    let col = &batch.columns[col_idx];
    if row_idx >= col.data.len() || col.nulls.is_null(row_idx) {
        return None;
    }
    match &col.data {
        ColumnData::Binary(blobs) if row_idx < blobs.len() => Some(&blobs[row_idx]),
        _ => None,
    }
}

/// Reinterprets a byte slice as a slice of f32 values. Each 4 bytes in
/// little-endian order represent one f32. Returns an empty slice if the
/// input length is not a multiple of 4.
fn bytes_to_f32_slice(bytes: &[u8]) -> &[f32] {
    if bytes.len() % 4 != 0 || bytes.is_empty() {
        return &[];
    }
    // The vector column stores raw f32 bytes in native endianness (LE on x86).
    // Alignment is guaranteed by Vec<u8> backing store on all supported platforms.
    let (prefix, floats, suffix) = unsafe { bytes.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        // Fallback: not aligned, should not happen in practice.
        return &[];
    }
    floats
}

/// Extracts a column's raw byte payload regardless of its declared type.
/// Used for spatial index maintenance to read the WKB-encoded geometry
/// payload out of either a Binary or Geometry-typed column.
pub fn extract_column_bytes<'a>(
    batch: &'a DataBatch,
    row_idx: usize,
    columns: &[zyron_catalog::ColumnEntry],
    target_column_id: zyron_catalog::ColumnId,
) -> Option<&'a [u8]> {
    let col_idx = columns.iter().position(|c| c.id == target_column_id)?;
    if col_idx >= batch.columns.len() {
        return None;
    }
    let col = &batch.columns[col_idx];
    if row_idx >= col.data.len() || col.nulls.is_null(row_idx) {
        return None;
    }
    match &col.data {
        ColumnData::Binary(blobs) if row_idx < blobs.len() => Some(&blobs[row_idx]),
        _ => None,
    }
}

// MBR computation from a decoded Geometry lives in zyron-types so startup
// recovery in zyron-server can call the same logic. See
// zyron_types::spatial_index::mbr_from_geometry.

/// Serializes a TupleId into bytes for WAL payload.
fn tuple_id_payload(tid: &TupleId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(14);
    buf.extend_from_slice(&tid.page_id.file_id.to_le_bytes());
    buf.extend_from_slice(&tid.page_id.page_num.to_le_bytes());
    buf.extend_from_slice(&tid.slot_id.to_le_bytes());
    buf
}

// ---------------------------------------------------------------------------
// Helper: build a single-row batch with the affected row count
// ---------------------------------------------------------------------------

pub(crate) fn count_batch(count: i64) -> DataBatch {
    let data = ColumnData::Int64(vec![count]);
    let nulls = NullBitmap::none(1);
    let col = Column::with_nulls(data, nulls, TypeId::Int64);
    DataBatch::new(vec![col])
}

// ---------------------------------------------------------------------------
// ValuesOperator
// ---------------------------------------------------------------------------

/// Produces rows from literal VALUES expressions.
/// Evaluates each row of expressions into a columnar batch.
pub struct ValuesOperator {
    rows: Vec<Vec<BoundExpr>>,
    schema: Vec<LogicalColumn>,
    /// Bound parameters from the extended query protocol. The single-row
    /// evaluator passes these to every `evaluate` call so VALUES expressions
    /// containing `$1`, `$2`, ... resolve correctly.
    params: Vec<ScalarValue>,
    /// Execution context for resolving sequence functions in VALUES rows.
    ctx: Option<Arc<ExecutionContext>>,
    /// True when any VALUES cell references a sequence function, computed once
    /// so the common path keeps its literal/parameter fast routes.
    has_sequence: bool,
    emitted: bool,
}

impl ValuesOperator {
    pub fn new(rows: Vec<Vec<BoundExpr>>, schema: Vec<LogicalColumn>) -> Self {
        let has_sequence = rows
            .iter()
            .any(|r| r.iter().any(crate::sequence::contains_sequence));
        Self {
            rows,
            schema,
            params: Vec::new(),
            ctx: None,
            has_sequence,
            emitted: false,
        }
    }

    /// Builds a values operator that evaluates against the given bound
    /// parameter set. Used by the extended query protocol to thread
    /// `$1`, `$2`, ... values into VALUES.
    pub fn with_params(
        rows: Vec<Vec<BoundExpr>>,
        schema: Vec<LogicalColumn>,
        params: Vec<ScalarValue>,
    ) -> Self {
        let has_sequence = rows
            .iter()
            .any(|r| r.iter().any(crate::sequence::contains_sequence));
        Self {
            rows,
            schema,
            params,
            ctx: None,
            has_sequence,
            emitted: false,
        }
    }

    /// Attaches the execution context so sequence functions in VALUES rows
    /// resolve against the catalog.
    pub fn with_context(mut self, ctx: Arc<ExecutionContext>) -> Self {
        self.ctx = Some(ctx);
        self
    }
}

impl Operator for ValuesOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.emitted || self.rows.is_empty() {
                return Ok(None);
            }
            self.emitted = true;

            let num_cols = self.schema.len();
            let num_rows = self.rows.len();

            let mut col_data: Vec<ColumnData> = self
                .schema
                .iter()
                .map(|c| {
                    ColumnData::with_capacity(
                        zyron_common::types::TypeId::timestamp_physical_type_id(
                            c.type_id,
                            c.ts_precision,
                        ),
                        num_rows,
                    )
                })
                .collect();
            let mut col_nulls: Vec<NullBitmap> =
                (0..num_cols).map(|_| NullBitmap::empty()).collect();

            // Single-row evaluation context for the per-row VALUES
            // expressions. It must report num_rows == 1 so literal columns are
            // length 1 (an empty batch would size every literal to 0 rows and
            // collapse all inserted values to NULL).
            let row_eval_ctx = DataBatch::new(vec![Column::new(
                ColumnData::Boolean(vec![false]),
                TypeId::Boolean,
            )]);

            for row_exprs in &self.rows {
                // When a row references sequence functions, resolve them once
                // per row against a single-row context so each VALUES row gets
                // its own nextval value, then evaluate the rewritten cells
                // against the extended batch.
                let mut seq_row: Option<(Vec<BoundExpr>, DataBatch, Vec<LogicalColumn>)> = None;
                if self.has_sequence {
                    if let Some(ctx) = &self.ctx {
                        let mut exprs = row_exprs.clone();
                        // A single-row, zero-column batch carries the row count.
                        // Sequence materialization appends one synthesized
                        // column per call; the batch and schema grow in lockstep
                        // so the synthesized column references resolve by
                        // position.
                        let mut batch = DataBatch {
                            columns: Vec::new(),
                            num_rows: 1,
                        };
                        let mut schema: Vec<LogicalColumn> = Vec::new();
                        crate::sequence::materialize_sequences(
                            &mut exprs,
                            &mut batch,
                            &mut schema,
                            ctx,
                        )
                        .await?;
                        seq_row = Some((exprs, batch, schema));
                    }
                }

                for (c, expr) in row_exprs.iter().enumerate() {
                    let entry = &self.schema[c];
                    let target = entry.type_id;

                    if let Some((exprs, batch, schema)) = &seq_row {
                        let col = evaluate(&exprs[c], batch, schema, &self.params)?;
                        let col = if col.len() > 0 && col.type_id != target && !col.is_null(0) {
                            crate::compute::cast_column(&col, target)?
                        } else {
                            col
                        };
                        let scalar = if col.len() > 0 {
                            col.get_scalar(0)
                        } else {
                            ScalarValue::Null
                        };
                        col_nulls[c].push(scalar.is_null());
                        col_data[c].push_scalar(&scalar);
                        continue;
                    }

                    // Fast paths produce a ScalarValue directly, skipping the
                    // per-cell Column alloc + cast_column dispatch the generic
                    // evaluator pays. Literal cells come from inline VALUES,
                    // Parameter cells come from a cached/parameterized plan
                    // (auto-param or extended protocol). Both fall back to the
                    // full evaluator for type combinations not covered by the
                    // fast coercions (Decimal, Interval, FixedBinary, etc.).
                    let fast = match expr {
                        BoundExpr::Literal { value, .. } => literal_to_scalar(value, target),
                        BoundExpr::Parameter { index, .. } => {
                            // 1-based parameter index into the bound values.
                            self.params
                                .get(index.wrapping_sub(1))
                                .and_then(|sv| crate::expr::coerce_scalar_to(sv, target))
                        }
                        _ => None,
                    };
                    let scalar = if let Some(s) = fast {
                        s
                    } else {
                        let col = evaluate(expr, &row_eval_ctx, &self.schema, &self.params)?;
                        let col = if col.len() > 0 && col.type_id != target && !col.is_null(0) {
                            crate::compute::cast_column(&col, target)?
                        } else {
                            col
                        };
                        if col.len() > 0 {
                            col.get_scalar(0)
                        } else {
                            ScalarValue::Null
                        }
                    };

                    // TIMESTAMP(p>6) columns store i128 picoseconds. A
                    // timestamp literal evaluates to i64 microseconds, scale
                    // it up exactly into the i128 buffer.
                    let scalar = if entry.ts_precision.unwrap_or(6) > 6
                        && matches!(target, TypeId::Timestamp | TypeId::TimestampTz)
                    {
                        match scalar {
                            ScalarValue::Int64(us) => ScalarValue::Int128(us as i128 * 1_000_000),
                            ScalarValue::Int128(ps) => ScalarValue::Int128(ps),
                            other => other,
                        }
                    } else {
                        scalar
                    };
                    col_nulls[c].push(scalar.is_null());
                    col_data[c].push_scalar(&scalar);
                }
            }

            let columns: Vec<Column> = col_data
                .into_iter()
                .zip(col_nulls)
                .zip(self.schema.iter())
                .map(|((data, nulls), lc)| {
                    Column::with_nulls_ts(data, nulls, lc.type_id, lc.ts_precision)
                })
                .collect();

            Ok(Some(ExecutionBatch::new(DataBatch::new(columns))))
        })
    }
}

// ---------------------------------------------------------------------------
// InsertOperator
// ---------------------------------------------------------------------------

/// Pulls rows from a source operator, encodes them as tuples,
/// logs to WAL, inserts into the heap file, and returns the row count.
pub struct InsertOperator {
    source: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    /// Column ids the source rows supply, in source-column order. The source
    /// batch is reshaped to full table-column order before encoding; columns
    /// the statement omitted are filled with their DEFAULT or NULL.
    target_columns: Vec<zyron_catalog::ColumnId>,
    /// Bound default expressions for omitted columns, keyed by column id.
    column_defaults: Vec<(zyron_catalog::ColumnId, zyron_planner::binder::BoundExpr)>,
    /// CHECK constraint predicates (bound at table_idx 0) enforced per row.
    check_constraints: Vec<zyron_planner::binder::BoundExpr>,
    /// Data-quality expectations (bound at table_idx 0) applied per row. Each
    /// carries a violation action: Fail aborts, Warn counts, Drop removes the
    /// row from the insert, Quarantine routes the row to a companion table.
    expectations: Vec<zyron_planner::binder::BoundExpectation>,
    finished: bool,
}

impl InsertOperator {
    pub fn new(
        source: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        target_columns: Vec<zyron_catalog::ColumnId>,
        column_defaults: Vec<(zyron_catalog::ColumnId, zyron_planner::binder::BoundExpr)>,
        check_constraints: Vec<zyron_planner::binder::BoundExpr>,
        expectations: Vec<zyron_planner::binder::BoundExpectation>,
    ) -> Self {
        Self {
            source,
            ctx,
            table_id,
            target_columns,
            column_defaults,
            check_constraints,
            expectations,
            finished: false,
        }
    }
}

/// Writes violating rows into a quarantine table. The rows are gathered from
/// the reshaped (table-column order) batch and two metadata columns are
/// appended: the name of the violated expectation and the capture time in
/// epoch microseconds. The quarantine table carries no indexes, triggers, or
/// CDC, so this writes the heap and WAL directly. Takes the context by Arc
/// reference so the future stays Send (the operator itself is not Sync).
async fn write_quarantine(
    ctx: &Arc<ExecutionContext>,
    quarantine_table_id: u32,
    batch: &DataBatch,
    rows: &[usize],
    names: &[String],
    txn_id: u32,
) -> zyron_common::Result<()> {
    let q_table_id = zyron_catalog::TableId(quarantine_table_id);
    let q_entry = ctx.get_table_entry(q_table_id)?;
    let q_heap = ctx.get_heap_file(q_table_id).await?;

    let indices: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
    let mut q_batch = batch.take(&indices);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64;
    q_batch.columns.push(Column::new(
        crate::column::ColumnData::Utf8(names.to_vec()),
        zyron_common::TypeId::Varchar,
    ));
    q_batch.columns.push(Column::new(
        crate::column::ColumnData::Int64(vec![now; indices.len()]),
        zyron_common::TypeId::Int64,
    ));

    let tuples = batch_to_tuples(&q_batch, &q_entry.columns, txn_id);
    let mut records: Vec<(u32, &[u8])> = Vec::with_capacity(tuples.len());
    for t in &tuples {
        records.push((txn_id, t.data()));
    }
    let last_lsn = ctx.wal.log_insert_batch_last_lsn(&records)?;
    ctx.mark_wrote_wal();
    let tuple_ids = q_heap.insert_batch(&tuples).await?;
    if let Some(first) = tuple_ids.first() {
        let mut prev_page = first.page_id;
        ctx.buffer_pool.mark_dirty_with_lsn(prev_page, last_lsn.0);
        for tid in tuple_ids.iter().skip(1) {
            if tid.page_id != prev_page {
                ctx.buffer_pool.mark_dirty_with_lsn(tid.page_id, last_lsn.0);
                prev_page = tid.page_id;
            }
        }
    }
    Ok(())
}

/// Writes a batch's foreign-key rejects into their quarantine tables and
/// returns the batch without them.
///
/// The rejected rows are preserved with the constraint that rejected them,
/// which is the point of quarantining rather than dropping: a bulk load
/// finishes and the rows it could not place are still there to fix.
async fn divert_quarantined(
    ctx: &Arc<ExecutionContext>,
    batch: &DataBatch,
    violations: &crate::operator::fk::FkViolations,
    txn_id: u32,
) -> zyron_common::Result<DataBatch> {
    for (quarantine_id, rows, names) in violations.by_table() {
        write_quarantine(ctx, quarantine_id, batch, &rows, &names, txn_id).await?;
    }
    let rejected = violations.rows();
    let keep: Vec<u32> = (0..batch.num_rows)
        .filter(|row| rejected.binary_search(row).is_err())
        .map(|row| row as u32)
        .collect();
    Ok(batch.take(&keep))
}

/// Per-batch disposition computed from a table's data-quality expectations.
struct ExpectationResult {
    /// Row kept in the main insert when true. Rows quarantined or dropped are
    /// set false.
    keep_mask: Vec<bool>,
    /// Quarantine routing grouped by target table id: (table_id, row indices,
    /// violated expectation name per row).
    quarantine: Vec<(u32, Vec<usize>, Vec<String>)>,
    /// Number of rows that violated a Warn expectation.
    warn_count: usize,
}

/// Evaluates a table's expectations over a reshaped (table-column order) batch
/// and computes each row's disposition. A predicate that is false is a
/// violation; NULL (unknown) passes, matching CHECK semantics. Fail aborts the
/// statement. Quarantine is evaluated before Drop so a row failing both is
/// preserved in the quarantine table rather than silently dropped.
fn evaluate_expectations(
    expectations: &[zyron_planner::binder::BoundExpectation],
    batch: &DataBatch,
    table_columns: &[zyron_catalog::ColumnEntry],
    params: &[crate::column::ScalarValue],
) -> zyron_common::Result<ExpectationResult> {
    use zyron_catalog::ExpectationAction;

    let n = batch.num_rows;
    let schema: Vec<LogicalColumn> = table_columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            ts_precision: c.ts_precision,
        })
        .collect();

    let mut keep_mask = vec![true; n];
    let mut warn_count = 0usize;
    let mut quarantine: Vec<(u32, Vec<usize>, Vec<String>)> = Vec::new();
    let mut q_index: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    let is_violation =
        |sv: crate::column::ScalarValue| matches!(sv, crate::column::ScalarValue::Boolean(false));

    // Fail predicates abort the statement on the first violating row.
    for exp in expectations {
        if exp.on_violation != ExpectationAction::Fail {
            continue;
        }
        let result = crate::expr::evaluate(&exp.predicate, batch, &schema, params)?;
        for row in 0..n {
            if is_violation(result.get_scalar(row)) {
                return Err(zyron_common::ZyronError::CheckViolation(format!(
                    "row {row} violates expectation \"{}\"",
                    exp.name
                )));
            }
        }
    }

    // Quarantine predicates route the first-matching violating row to a target
    // table and remove it from the main insert.
    for exp in expectations {
        if exp.on_violation != ExpectationAction::Quarantine {
            continue;
        }
        let Some(qid) = exp.quarantine_table_id else {
            continue;
        };
        let result = crate::expr::evaluate(&exp.predicate, batch, &schema, params)?;
        for row in 0..n {
            if !keep_mask[row] {
                continue;
            }
            if is_violation(result.get_scalar(row)) {
                keep_mask[row] = false;
                let idx = *q_index.entry(qid).or_insert_with(|| {
                    quarantine.push((qid, Vec::new(), Vec::new()));
                    quarantine.len() - 1
                });
                quarantine[idx].1.push(row);
                quarantine[idx].2.push(exp.name.clone());
            }
        }
    }

    // Drop predicates remove violating rows from the main insert.
    for exp in expectations {
        if exp.on_violation != ExpectationAction::Drop {
            continue;
        }
        let result = crate::expr::evaluate(&exp.predicate, batch, &schema, params)?;
        for row in 0..n {
            if !keep_mask[row] {
                continue;
            }
            if is_violation(result.get_scalar(row)) {
                keep_mask[row] = false;
            }
        }
    }

    // Warn predicates count violations and keep the rows.
    for exp in expectations {
        if exp.on_violation != ExpectationAction::Warn {
            continue;
        }
        let result = crate::expr::evaluate(&exp.predicate, batch, &schema, params)?;
        for row in 0..n {
            if is_violation(result.get_scalar(row)) {
                warn_count += 1;
            }
        }
    }

    Ok(ExpectationResult {
        keep_mask,
        quarantine,
        warn_count,
    })
}

/// Evaluates each CHECK predicate over a table-column-order batch and rejects
/// the statement if any row makes a predicate false. A predicate that is NULL
/// (unknown) passes, matching SQL semantics. The predicates were bound against
/// the table at table_idx 0, so the schema is rebuilt at that index here.
/// Rejects a vector whose length is not the column's declared dimension.
///
/// A vector column's dimension is what the index is built for and what every
/// distance is computed over, so a row of the wrong length would score
/// against unrelated components rather than fail. The check reads the packed
/// byte length, which is four bytes per component, so it costs nothing per
/// row beyond a comparison.
pub(crate) fn enforce_vector_dimensions(
    batch: &DataBatch,
    table_columns: &[zyron_catalog::ColumnEntry],
) -> zyron_common::Result<()> {
    if batch.num_rows == 0 {
        return Ok(());
    }
    for (idx, column) in table_columns.iter().enumerate() {
        if column.type_id != zyron_common::TypeId::Vector {
            continue;
        }
        let Some(dimensions) = column.max_length.filter(|n| *n > 0) else {
            continue;
        };
        let Some(data) = batch.columns.get(idx) else {
            continue;
        };
        let expected = dimensions * 4;
        for row in 0..batch.num_rows.min(data.len()) {
            if data.is_null(row) {
                continue;
            }
            let crate::column::ScalarValue::Binary(bytes) = data.get_scalar(row) else {
                continue;
            };
            if bytes.len() != expected {
                return Err(zyron_common::ZyronError::ExecutionError(format!(
                    "column \"{}\" holds vectors of {} components, row {} has {}",
                    column.name,
                    dimensions,
                    row,
                    bytes.len() / 4
                )));
            }
        }
    }
    Ok(())
}

/// Re-encodes array values to the element type their column declares.
///
/// An array constructor takes its element type from the expressions in it,
/// so `ARRAY[10, 20]` carries whatever an integer literal binds to. Writing
/// that into an `INT[]` column unchanged would store elements twice as wide
/// as the declaration asks for. A value already at the declared type is left
/// alone, so the pass costs one header read per array value in the common
/// case and touches no other column.
pub(crate) fn normalize_array_elements(
    batch: &mut DataBatch,
    table_columns: &[zyron_catalog::ColumnEntry],
) -> zyron_common::Result<()> {
    if batch.num_rows == 0 {
        return Ok(());
    }
    for (idx, column) in table_columns.iter().enumerate() {
        let Some(declared) = column.element_type else {
            continue;
        };
        if column.type_id != zyron_common::TypeId::Array {
            continue;
        }
        let Some(data) = batch.columns.get_mut(idx) else {
            continue;
        };
        let ColumnData::Binary(values) = &mut data.data else {
            continue;
        };
        let width = declared.fixed_size().unwrap_or(0);
        for row in 0..values.len() {
            let Some(view) = zyron_common::ArrayView::parse(&values[row]) else {
                continue;
            };
            if view.element_type() == declared {
                continue;
            }
            let source_type = view.element_type();
            let source_width = source_type.fixed_size().unwrap_or(0);
            let mut payloads: Vec<Option<Vec<u8>>> = Vec::with_capacity(view.len());
            for element in view.iter() {
                let Some(bytes) = element else {
                    payloads.push(None);
                    continue;
                };
                let scalar = if source_width > 0 {
                    crate::batch::decode_fixed_scalar(source_type, bytes)
                } else {
                    crate::batch::decode_varlen_scalar(source_type, bytes)
                };
                let converted = crate::compute::cast_scalar(&scalar, declared)?;
                payloads.push(Some(crate::batch::encode_scalar_value(
                    declared, &converted, width,
                )));
            }
            let borrowed: Vec<Option<&[u8]>> = payloads.iter().map(|p| p.as_deref()).collect();
            values[row] = zyron_common::array_value::encode(declared, &borrowed);
        }
    }
    Ok(())
}

/// Rejects a null in a column declared NOT NULL.
///
/// A declared constraint that does nothing is a lie, and NOT NULL is the one
/// every column carries, so it is checked on the full-width row image that
/// every write path builds. The cost is a bitmap test per non-nullable
/// column: a batch with no nulls answers without reading a row.
pub(crate) fn enforce_not_null(
    batch: &DataBatch,
    table_columns: &[zyron_catalog::ColumnEntry],
) -> zyron_common::Result<()> {
    for (ci, col_entry) in table_columns.iter().enumerate() {
        if col_entry.nullable {
            continue;
        }
        let Some(column) = batch.columns.get(ci) else {
            continue;
        };
        if !column.nulls.has_nulls() {
            continue;
        }
        for row in 0..batch.num_rows {
            if column.is_null(row) {
                return Err(zyron_common::ZyronError::CheckViolation(format!(
                    "null value in column \"{}\" violates NOT NULL",
                    col_entry.name
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn enforce_check_constraints(
    checks: &[zyron_planner::binder::BoundExpr],
    batch: &DataBatch,
    table_columns: &[zyron_catalog::ColumnEntry],
    params: &[crate::column::ScalarValue],
) -> zyron_common::Result<()> {
    enforce_not_null(batch, table_columns)?;
    enforce_vector_dimensions(batch, table_columns)?;
    if checks.is_empty() || batch.num_rows == 0 {
        return Ok(());
    }
    let schema: Vec<LogicalColumn> = table_columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            ts_precision: c.ts_precision,
        })
        .collect();
    for expr in checks {
        let result = crate::expr::evaluate(expr, batch, &schema, params)?;
        for row in 0..batch.num_rows {
            if matches!(
                result.get_scalar(row),
                crate::column::ScalarValue::Boolean(false)
            ) {
                return Err(zyron_common::ZyronError::CheckViolation(format!(
                    "row {row} violates a CHECK constraint"
                )));
            }
        }
    }
    Ok(())
}

/// Reshapes a source batch (columns in `target_columns` order) into full
/// table-column order. A column the INSERT omitted is filled with its bound
/// DEFAULT expression when one exists, otherwise NULL. The caller skips this
/// entirely for the identity mapping (source already in full table order).
fn reshape_insert_batch(
    batch: DataBatch,
    plan: &[Option<usize>],
    table_columns: &[zyron_catalog::ColumnEntry],
    defaults: &[Option<zyron_planner::binder::BoundExpr>],
    seq_defaults: &mut [Option<crate::column::Column>],
    params: &[crate::column::ScalarValue],
) -> zyron_common::Result<DataBatch> {
    let num_rows = batch.num_rows;
    let mut source: Vec<Option<crate::column::Column>> =
        batch.columns.into_iter().map(Some).collect();
    // Defaults reference no columns, so an empty schema and a row-count-carrying
    // batch are enough to broadcast a constant or volatile value over the rows.
    let row_batch = DataBatch {
        columns: Vec::new(),
        num_rows,
    };
    let mut columns: Vec<crate::column::Column> = Vec::with_capacity(plan.len());
    for (i, src_pos) in plan.iter().enumerate() {
        let col = match src_pos {
            // Each source column maps to exactly one table column, so take it.
            Some(j) => source[*j].take().expect("source column mapped once"),
            None => {
                // A sequence-backed DEFAULT is pre-materialized into a per-row
                // column for this batch; use it when present.
                if let Some(seq_col) = seq_defaults.get_mut(i).and_then(Option::take) {
                    crate::compute::cast_column(&seq_col, table_columns[i].type_id)?
                } else {
                    match &defaults[i] {
                        Some(expr) => {
                            let evaluated = crate::expr::evaluate(expr, &row_batch, &[], params)?;
                            crate::compute::cast_column(&evaluated, table_columns[i].type_id)?
                        }
                        None => {
                            crate::column::Column::null_column(table_columns[i].type_id, num_rows)
                        }
                    }
                }
            }
        };
        columns.push(col);
    }
    Ok(DataBatch::new(columns))
}

/// Materializes a DEFAULT expression that references a sequence function into
/// a column of `num_rows` values. nextval produces a distinct value per row.
async fn eval_sequence_default(
    expr: &zyron_planner::binder::BoundExpr,
    num_rows: usize,
    ctx: &ExecutionContext,
) -> zyron_common::Result<crate::column::Column> {
    let mut exprs = [expr.clone()];
    let mut batch = DataBatch {
        columns: Vec::new(),
        num_rows,
    };
    let mut schema: Vec<LogicalColumn> = Vec::new();
    crate::sequence::materialize_sequences(&mut exprs, &mut batch, &mut schema, ctx).await?;
    crate::expr::evaluate(&exprs[0], &batch, &schema, &ctx.params)
}

impl Operator for InsertOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;

            self.ctx.ensure_writable("INSERT")?;

            let table_entry = self.ctx.get_table_entry(self.table_id)?;
            let is_lake = table_entry.lake.is_lake();
            // A lake table never opens its allocated heap file, resolving
            // the handle would materialize an empty heap file on disk
            let heap_file = if is_lake {
                None
            } else {
                self.ctx
                    .ensure_heap_branch_resolved("INSERT", &table_entry.name)?;
                Some(self.ctx.get_heap_file(self.table_id).await?)
            };
            let mut total_inserted: i64 = 0;
            let mut lake_batches: Vec<DataBatch> = Vec::new();
            let txn_id = self.ctx.txn_id;

            // Map each table column to the source-batch position that supplies
            // it (None = omitted). When the source already produces every table
            // column in order, skip reshaping entirely. Omitted columns are
            // filled from `reshape_defaults` (their bound DEFAULT or NULL).
            let reshape_plan: Option<Vec<Option<usize>>> = {
                let identity = self.target_columns.len() == table_entry.columns.len()
                    && self
                        .target_columns
                        .iter()
                        .zip(table_entry.columns.iter())
                        .all(|(tid, c)| *tid == c.id);
                if identity {
                    None
                } else {
                    let mut pos_by_id: std::collections::HashMap<zyron_catalog::ColumnId, usize> =
                        std::collections::HashMap::with_capacity(self.target_columns.len());
                    for (j, cid) in self.target_columns.iter().enumerate() {
                        pos_by_id.insert(*cid, j);
                    }
                    Some(
                        table_entry
                            .columns
                            .iter()
                            .map(|c| pos_by_id.get(&c.id).copied())
                            .collect(),
                    )
                }
            };
            // Per-table-column default expression, aligned to table column order.
            let reshape_defaults: Vec<Option<zyron_planner::binder::BoundExpr>> =
                if reshape_plan.is_some() {
                    table_entry
                        .columns
                        .iter()
                        .map(|c| {
                            self.column_defaults
                                .iter()
                                .find(|(id, _)| *id == c.id)
                                .map(|(_, expr)| expr.clone())
                        })
                        .collect()
                } else {
                    Vec::new()
                };
            // Table-column positions whose DEFAULT references a sequence
            // function. These are resolved per batch into per-row columns
            // (nextval yields a distinct value per row) before reshaping.
            let seq_default_positions: Vec<usize> = reshape_defaults
                .iter()
                .enumerate()
                .filter_map(|(i, d)| match d {
                    Some(expr) if crate::sequence::contains_sequence(expr) => Some(i),
                    _ => None,
                })
                .collect();
            let params = self.ctx.params.clone();

            // Resolve the index snapshot ONCE per statement instead of per
            // batch. Pre-bind the per-index handles so the inner row loop
            // only does work proportional to actual indexes, not catalog
            // lookups.
            let index_snap = self.ctx.index_snapshot_for_table(self.table_id.0);
            let fts_resolved: Vec<(zyron_catalog::IndexId, Arc<zyron_search::InvertedIndex>)> =
                if index_snap.fts.is_empty() {
                    Vec::new()
                } else if let Some(mgr) = self.ctx.fts_manager.as_ref() {
                    index_snap
                        .fts
                        .iter()
                        .filter_map(|id| mgr.get_index(id.0).map(|idx| (*id, idx)))
                        .collect()
                } else {
                    Vec::new()
                };
            let vec_resolved: Vec<(u32, Arc<zyron_search::vector::VectorIndex>)> =
                if index_snap.vector.is_empty() {
                    Vec::new()
                } else {
                    index_snap
                        .vector
                        .iter()
                        .filter_map(|id| self.ctx.get_vector_index(id.0).map(|idx| (id.0, idx)))
                        .collect()
                };

            loop {
                self.ctx.check_cancelled()?;
                let input = self.source.next().await?;
                let Some(mut exec_batch) = input else {
                    break;
                };

                // Reshape a partial or reordered column list into full
                // table-column order before any encoding or constraint check,
                // filling omitted columns from their DEFAULT or NULL.
                if let Some(plan) = &reshape_plan {
                    // Materialize sequence-backed DEFAULTs into per-row columns
                    // for this batch. nextval allocates one value per row.
                    let mut seq_default_cols: Vec<Option<Column>> =
                        vec![None; table_entry.columns.len()];
                    if !seq_default_positions.is_empty() {
                        let num_rows = exec_batch.batch.num_rows;
                        for &i in &seq_default_positions {
                            if let Some(expr) = &reshape_defaults[i] {
                                seq_default_cols[i] =
                                    Some(eval_sequence_default(expr, num_rows, &self.ctx).await?);
                            }
                        }
                    }
                    exec_batch.batch = reshape_insert_batch(
                        exec_batch.batch,
                        plan,
                        &table_entry.columns,
                        &reshape_defaults,
                        &mut seq_default_cols,
                        &params,
                    )?;
                }

                // Apply data-quality expectations before any integrity check or
                // write. Fail aborts the statement, Quarantine routes violating
                // rows to a companion table, Drop removes them, Warn counts
                // them. The surviving rows flow into the checks and write below.
                if !self.expectations.is_empty() {
                    let outcome = evaluate_expectations(
                        &self.expectations,
                        &exec_batch.batch,
                        &table_entry.columns,
                        &params,
                    )?;
                    if outcome.warn_count > 0 {
                        eprintln!(
                            "expectation warning: {} row(s) on table {} failed a WARN expectation",
                            outcome.warn_count, table_entry.name
                        );
                    }
                    for (qid, rows, names) in &outcome.quarantine {
                        if rows.is_empty() {
                            continue;
                        }
                        write_quarantine(&self.ctx, *qid, &exec_batch.batch, rows, names, txn_id)
                            .await?;
                    }
                    if outcome.keep_mask.iter().any(|&keep| !keep) {
                        exec_batch.batch = exec_batch.batch.filter(&outcome.keep_mask);
                    }
                    if exec_batch.batch.num_rows == 0 {
                        continue;
                    }
                }

                // Arrays take the element width their column declares before
                // any check reads the row, so a CHECK sees the stored image
                normalize_array_elements(&mut exec_batch.batch, &table_entry.columns)?;

                // Enforce CHECK constraints on the full-width row image before
                // any write so a violation aborts the statement with no effect.
                enforce_check_constraints(
                    &self.check_constraints,
                    &exec_batch.batch,
                    &table_entry.columns,
                    &params,
                )?;

                // A lake table's rows go to its transaction log, never to
                // heap pages. The reshaped, defaulted, checked batch is
                // accumulated and committed as one data file per statement
                // after the source drains
                if table_entry.lake.is_lake() {
                    // BEFORE INSERT fires on the reshaped, defaulted image,
                    // the same one the heap path shows its triggers, and
                    // before any constraint reads it so a trigger that
                    // rejects a row stops it reaching storage
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_BEFORE,
                        zyron_catalog::TriggerEntry::EVENT_INSERT,
                        &exec_batch.batch,
                        &table_entry.columns,
                    )
                    .await?;

                    // Referential integrity runs per batch, against a heap
                    // or a lake parent, before the rows are accumulated. A
                    // constraint that quarantines diverts its rejected rows
                    // and the rest of the batch still lands
                    let violations = crate::operator::fk::check_child_fks(
                        &self.ctx,
                        &table_entry,
                        &exec_batch.batch,
                    )
                    .await?;
                    let batch = if violations.is_empty() {
                        exec_batch.batch
                    } else {
                        divert_quarantined(&self.ctx, &exec_batch.batch, &violations, txn_id as u32)
                            .await?
                    };
                    if batch.num_rows == 0 {
                        continue;
                    }
                    total_inserted += batch.num_rows as i64;
                    lake_batches.push(batch);
                    continue;
                }
                let heap_file = heap_file.as_ref().ok_or_else(|| {
                    zyron_common::ZyronError::ExecutionError(
                        "heap insert path reached without a heap file".into(),
                    )
                })?;

                // Branch writes go to the branch append overlay and stay
                // isolated from the main line. Index, CDC, and trigger
                // maintenance for branches arrives in a later stage.
                if let Some(bid) = self.ctx.active_branch_id {
                    let tuples = batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                    let ids = crate::operator::branch_write::branch_insert(
                        &self.ctx,
                        bid,
                        heap_file.heap_file_id(),
                        &tuples,
                    )
                    .await?;
                    total_inserted += ids.len() as i64;
                    continue;
                }

                // Enforce child-side foreign keys before any write so a
                // violation aborts the statement without partial effects. A
                // constraint that quarantines diverts its rejected rows here
                // instead, leaving the rest of the batch to insert
                let fk_violations = crate::operator::fk::check_child_fks(
                    &self.ctx,
                    &table_entry,
                    &exec_batch.batch,
                )
                .await?;
                if !fk_violations.is_empty() {
                    exec_batch.batch = divert_quarantined(
                        &self.ctx,
                        &exec_batch.batch,
                        &fk_violations,
                        txn_id as u32,
                    )
                    .await?;
                    if exec_batch.batch.num_rows == 0 {
                        continue;
                    }
                }

                // Enforce unique constraints before any write. Abort does not
                // undo heap rows here, so the check must precede the mutation.
                #[cfg(feature = "profile")]
                let _uc_span =
                    zyron_common::profile::scope(zyron_common::profile::Phase::ExecUniqueCheck);
                check_unique_constraints(
                    &self.ctx,
                    &table_entry,
                    &exec_batch.batch,
                    &index_snap,
                    &[],
                )
                .await?;
                #[cfg(feature = "profile")]
                drop(_uc_span);

                let tuples = batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);

                // Reused scratch lives outside the inner alloc paths so the
                // common-case OLTP single-row insert does not heap-allocate
                // a fresh Vec for the trigger payload, the WAL record list,
                // or the dirty-page set on every call.
                let mut batch_records: Vec<(u32, &[u8])> = Vec::with_capacity(tuples.len());
                for t in &tuples {
                    batch_records.push((txn_id, t.data()));
                }

                // Fire BEFORE INSERT triggers if present.
                if let Some(ref hook) = self.ctx.dml_hook {
                    let mut tuple_refs: Vec<&[u8]> = Vec::with_capacity(tuples.len());
                    for t in &tuples {
                        tuple_refs.push(t.data());
                    }
                    if !hook.before_insert(self.table_id.0, &tuple_refs, txn_id)? {
                        continue; // Trigger cancelled the insert
                    }
                }

                // Fire BEFORE INSERT row/statement triggers in the same txn.
                crate::trigger::fire_row_triggers(
                    &self.ctx,
                    self.table_id,
                    zyron_catalog::TriggerEntry::TIMING_BEFORE,
                    zyron_catalog::TriggerEntry::EVENT_INSERT,
                    &exec_batch.batch,
                    &table_entry.columns,
                )
                .await?;

                // Batch WAL log: one CAS + commit for all inserts in this batch
                // Use the last-LSN-only variant so the WAL writer skips its
                // per-record Vec<Lsn> allocation, callers further down the
                // pipeline only need the last LSN to chain to the Commit record
                let last_lsn = self.ctx.wal.log_insert_batch_last_lsn(&batch_records)?;
                self.ctx.mark_wrote_wal();

                #[cfg(feature = "profile")]
                let _heap_span =
                    zyron_common::profile::scope(zyron_common::profile::Phase::ExecHeapInsert);
                let tuple_ids = heap_file.insert_batch(&tuples).await?;
                #[cfg(feature = "profile")]
                drop(_heap_span);

                // Drop any graph CSR built from this table so the next graph
                // algorithm rebuilds from current data.
                if let Some(gm) = &self.ctx.graph_manager {
                    gm.invalidate_for_table(self.table_id.0);
                }

                // Stamp dirty pages with WAL LSN for checkpoint ordering.
                // Walk tuple_ids in order and emit one mark_dirty per distinct
                // page, since insert_batch writes to consecutive heap pages
                // and the buffer-pool dirty stamp is a per-page atomic. This
                // turns an O(rows) hash lookup into O(unique pages).
                if let Some(first) = tuple_ids.first() {
                    let mut prev_page = first.page_id;
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(prev_page, last_lsn.0);
                    for tid in tuple_ids.iter().skip(1) {
                        if tid.page_id != prev_page {
                            self.ctx
                                .buffer_pool
                                .mark_dirty_with_lsn(tid.page_id, last_lsn.0);
                            prev_page = tid.page_id;
                        }
                    }
                }

                total_inserted += tuples.len() as i64;

                // Record a reverse-insert per row so ROLLBACK TO SAVEPOINT
                // self-deletes these rows. Gated on an open savepoint, so the
                // common path records nothing.
                if self.ctx.recording_undo() {
                    for tid in &tuple_ids {
                        self.ctx.record_insert_undo(
                            table_entry.heap_file_id,
                            table_entry.fsm_file_id,
                            *tid,
                        );
                    }
                }

                // One registry ordinal per inserted row, shared by FTS,
                // vector and spatial maintenance below. Allocated only when
                // a search index exists, ordinary tables pay nothing.
                let needs_docs = !fts_resolved.is_empty()
                    || !vec_resolved.is_empty()
                    || !index_snap.spatial.is_empty();
                let doc_ids: Vec<u64> = if needs_docs {
                    let Some(reg) = &self.ctx.doc_registry else {
                        return Err(ZyronError::Internal(
                            "search index maintenance requires the document registry".into(),
                        ));
                    };
                    tuple_ids
                        .iter()
                        .map(|tid| reg.allocate(table_entry.id.0, tid.locator()))
                        .collect()
                } else {
                    Vec::new()
                };

                // Maintain FTS indexes: add each inserted document.
                let fts_indexes: &[(zyron_catalog::IndexId, Arc<zyron_search::InvertedIndex>)] =
                    fts_resolved.as_slice();
                if !fts_indexes.is_empty() {
                    let analyzer = zyron_search::SimpleAnalyzer;
                    let mut fts_buf = zyron_search::AnalysisBuffer::new();
                    let mut text_buf = String::with_capacity(256);
                    for (row_idx, _tid) in tuple_ids.iter().enumerate() {
                        let doc_id = doc_ids[row_idx];
                        text_buf.clear();
                        extract_fts_text_into(
                            &exec_batch.batch,
                            row_idx,
                            &table_entry.columns,
                            &mut text_buf,
                        );
                        for (idx_id, fts_idx) in fts_indexes.iter() {
                            if let Err(e) = fts_idx.add_document_with_buf(
                                doc_id,
                                &text_buf,
                                &analyzer,
                                &mut fts_buf,
                            ) {
                                eprintln!("FTS index {} insert failed: {e}", idx_id.0);
                            }
                        }
                    }
                }

                // Maintain vector indexes: insert each new vector into every
                // vector index on the table, sourced from that index's column.
                if !vec_resolved.is_empty() {
                    for (row_idx, _tid) in tuple_ids.iter().enumerate() {
                        let vec_id = doc_ids[row_idx];
                        for (idx_id, vec_idx) in &vec_resolved {
                            let col_id = vec_idx.column_id();
                            if let Some(vec_bytes) = extract_vector_bytes(
                                &exec_batch.batch,
                                row_idx,
                                &table_entry.columns,
                                col_id,
                            ) {
                                let vec_data = bytes_to_f32_slice(vec_bytes);
                                if let Err(e) = zyron_search::vector::VectorSearch::insert(
                                    vec_idx.as_ref(),
                                    vec_id,
                                    vec_data,
                                ) {
                                    eprintln!("vector index {} insert failed: {e}", idx_id);
                                }
                            }
                        }
                    }
                }

                // Maintain spatial (R-tree) indexes: for each indexed
                // geometry column, decode WKB to a Geometry, take its MBR,
                // and insert (mbr, rowid) into the live R-tree.
                if !index_snap.spatial.is_empty() {
                    if let Some(ref spatial_mgr) = self.ctx.spatial_manager {
                        for (row_idx, _tid) in tuple_ids.iter().enumerate() {
                            let rowid = doc_ids[row_idx];
                            for (idx_id, col_id) in &index_snap.spatial {
                                let Some(tree) = spatial_mgr.get(idx_id.0) else {
                                    continue;
                                };
                                let Some(geom_bytes) = extract_column_bytes(
                                    &exec_batch.batch,
                                    row_idx,
                                    &table_entry.columns,
                                    *col_id,
                                ) else {
                                    continue;
                                };
                                let Ok(geom) = zyron_types::geospatial::decode_wkb(geom_bytes)
                                else {
                                    continue;
                                };
                                let mbr = zyron_types::spatial_index::mbr_from_geometry(
                                    &geom,
                                    tree.dims(),
                                );
                                tree.insert(zyron_types::spatial_index::LeafEntry {
                                    mbr,
                                    data: rowid,
                                    deleted: false,
                                });
                            }
                        }
                    }
                }

                // Maintain B+Tree indexes for the inserted rows.
                #[cfg(feature = "profile")]
                let _idx_span =
                    zyron_common::profile::scope(zyron_common::profile::Phase::ExecIndexInsert);
                let index_locators: Vec<zyron_common::RowLocator> =
                    tuple_ids.iter().map(|t| t.locator()).collect();
                maintain_btree_insert(
                    &self.ctx,
                    &table_entry,
                    &exec_batch.batch,
                    &index_locators,
                    &index_snap,
                );
                #[cfg(feature = "profile")]
                drop(_idx_span);

                // Notify CDC hook if present.
                if let Some(ref hook) = self.ctx.cdc_hook {
                    let tuple_refs: Vec<&[u8]> = tuples.iter().map(|t| t.data()).collect();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as i64;
                    if let Err(e) =
                        hook.on_insert(self.table_id.0, &tuple_refs, last_lsn.0, now, txn_id, true)
                    {
                        eprintln!("CDC insert hook failed: {e}");
                    }
                }

                // Fire AFTER INSERT row/statement triggers in the same txn.
                crate::trigger::fire_row_triggers(
                    &self.ctx,
                    self.table_id,
                    zyron_catalog::TriggerEntry::TIMING_AFTER,
                    zyron_catalog::TriggerEntry::EVENT_INSERT,
                    &exec_batch.batch,
                    &table_entry.columns,
                )
                .await?;
            }

            // One data file and one log commit per statement, pending under
            // this transaction until its commit record is durable
            if is_lake && !lake_batches.is_empty() {
                let outcome = append_lake_batches(&self.ctx, &table_entry, &lake_batches)?;
                self.ctx.mark_wrote_wal();
                // The rows only have addresses once the commit assigned
                // them, so search index maintenance runs after the append
                // rather than per batch the way the heap path does
                if let Some(outcome) = outcome {
                    maintain_lake_search_indexes(
                        &self.ctx,
                        &table_entry,
                        &lake_batches,
                        outcome.partition_id,
                        &outcome.order,
                    )?;
                }
                // AFTER INSERT fires once the rows are committed, because a
                // trigger body that reads the table has to see them. One
                // firing per accumulated batch, so a trigger observes the
                // same batches its BEFORE counterpart did
                for batch in &lake_batches {
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_AFTER,
                        zyron_catalog::TriggerEntry::EVENT_INSERT,
                        batch,
                        &table_entry.columns,
                    )
                    .await?;
                }
            }

            if let Some(stats) = self.ctx.table_io_stats_for(self.table_id.0) {
                stats.record_inserts(total_inserted.max(0) as u64);
            }

            Ok(Some(ExecutionBatch::new(count_batch(total_inserted))))
        })
    }
}

/// Enforces every declared PRIMARY KEY and UNIQUE constraint against the
/// incoming batch and the table's live rows.
///
/// A declared constraint that does nothing is a lie, so these run by
/// default. The cost stays off the write path because the check reads the
/// manifest's per-file bounds and value blooms first: a monotonic key opens
/// one file and a key outside every stored range opens none.
pub(crate) fn enforce_lake_unique(
    log: &zyron_lake::TransactionLog,
    table_entry: &zyron_catalog::TableEntry,
    columns: &[zyron_lake::ColumnData],
    // Rows this statement removes, so an update keeping a key does not
    // collide with the copy it is rewriting. None for an insert, which
    // supersedes nothing
    superseded: Option<&zyron_lake::LakePredicate>,
) -> zyron_common::Result<()> {
    let mut specs: Vec<zyron_lake::UniqueSpec> = table_entry
        .constraints
        .iter()
        .filter(|c| {
            matches!(
                c.constraint_type,
                zyron_catalog::schema::ConstraintType::PrimaryKey
                    | zyron_catalog::schema::ConstraintType::Unique
            ) && !c.columns.is_empty()
                // NOT ENFORCED keeps the declaration for the planner and
                // asks the write path to skip it
                && c.enforced
        })
        .map(|c| zyron_lake::UniqueSpec {
            name: c.name.clone(),
            column_ids: c.columns.iter().map(|id| id.0 as u32).collect(),
        })
        .collect();
    let manifest = log.latest_manifest()?;
    // A unique index is a declaration too, and one that admits duplicates
    // is the same lie a constraint that does nothing would be. The lake's
    // own index specs carry the flag and the key columns, so they enforce
    // beside the constraints rather than through a second mechanism
    for index in &manifest.indexes {
        if !index.unique {
            continue;
        }
        if specs
            .iter()
            .any(|s| s.column_ids == index.column_ids)
        {
            continue;
        }
        specs.push(zyron_lake::UniqueSpec {
            name: index.name.clone(),
            column_ids: index.column_ids.clone(),
        });
    }
    if specs.is_empty() {
        return Ok(());
    }
    for spec in &specs {
        let (outcome, _) = zyron_lake::check_unique_replacing(
            log.paths(),
            &manifest,
            spec,
            columns,
            superseded,
        )?;
        match outcome {
            zyron_lake::UniqueOutcome::Ok => {}
            zyron_lake::UniqueOutcome::DuplicateInBatch {
                first_row,
                second_row,
            } => {
                return Err(zyron_common::ZyronError::ExecutionError(format!(
                    "duplicate key value violates unique constraint \"{}\" on \"{}\": \
                     inserted rows {} and {} carry the same key",
                    spec.name, table_entry.name, first_row, second_row
                )));
            }
            zyron_lake::UniqueOutcome::DuplicateWithStored { .. } => {
                return Err(zyron_common::ZyronError::ExecutionError(format!(
                    "duplicate key value violates unique constraint \"{}\" on \"{}\"",
                    spec.name, table_entry.name
                )));
            }
        }
    }
    Ok(())
}

/// Turns accumulated full-width insert batches into one lake data file
/// plus one Append commit, registered pending under the statement's
/// transaction. Cells use the storage representation, fixed little endian
/// padded to the physical width, variable length as bare bytes
fn append_lake_batches(
    ctx: &Arc<ExecutionContext>,
    table_entry: &zyron_catalog::TableEntry,
    batches: &[DataBatch],
) -> zyron_common::Result<Option<zyron_lake::AppendOutcome>> {
    let mut columns: Vec<zyron_lake::ColumnData> = table_entry
        .columns
        .iter()
        .map(|c| zyron_lake::ColumnData {
            column_id: c.id.0 as u32,
            cells: Vec::new(),
        })
        .collect();
    for batch in batches {
        for (ci, col_entry) in table_entry.columns.iter().enumerate() {
            let value_size = col_entry.physical_type_id().fixed_size().unwrap_or(0);
            let column = &batch.columns[ci];
            for r in 0..batch.num_rows {
                let sv = column.get_scalar(r);
                let cell = match sv {
                    ScalarValue::Null => None,
                    ref v => Some(crate::batch::encode_scalar_value(
                        col_entry.type_id,
                        v,
                        value_size,
                    )),
                };
                columns[ci].cells.push(cell);
            }
        }
    }
    if columns.first().map(|c| c.cells.is_empty()).unwrap_or(true) {
        return Ok(None);
    }
    let paths = zyron_lake::LakePaths::new(ctx.disk_manager.data_dir(), table_entry.id.0);
    // The branch this session writes, forked here if it has not touched
    // this table yet. Uniqueness reads the same head, so a key the branch
    // holds collides and a key only main holds does not
    let head = crate::operator::lake_scan::effective_head(ctx, None);
    let log = crate::operator::lake_scan::open_lake_write_head(&paths, &table_entry.name, head)?;
    let root = log.registry_key();
    enforce_lake_unique(&log, table_entry, &columns, None)?;
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::Append,
        db_txn_id: ctx.lake_txn_id(),
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        audit: None,
    };
    let out = zyron_lake::append_rows(&log, attempt, table_entry.id.0 as u64, &columns)?;
    zyron_lake::register_txn_pending(ctx.disk_manager.data_dir(), ctx.lake_txn_id(), root, out.version);
    Ok(Some(out))
}

/// Adds the rows one lake append wrote to the table's search indexes.
///
/// A lake row is addressed by data file and ordinal, and the writer sorts
/// the batch, so an input row's position is not its address. The append
/// returns the permutation and this inverts it, then allocates one registry
/// ordinal per row exactly as the heap path does, so FTS, vector and
/// spatial address a lake row through the same document identity they use
/// for every other store
pub(crate) fn maintain_lake_search_indexes(
    ctx: &Arc<ExecutionContext>,
    table_entry: &zyron_catalog::TableEntry,
    batches: &[DataBatch],
    partition_id: u64,
    order: &[usize],
) -> zyron_common::Result<()> {
    let index_snap = ctx.index_snapshot_for_table(table_entry.id.0);
    let fts_resolved: Vec<(zyron_catalog::IndexId, Arc<zyron_search::InvertedIndex>)> =
        if index_snap.fts.is_empty() {
            Vec::new()
        } else if let Some(mgr) = ctx.fts_manager.as_ref() {
            index_snap
                .fts
                .iter()
                .filter_map(|id| mgr.get_index(id.0).map(|idx| (*id, idx)))
                .collect()
        } else {
            Vec::new()
        };
    let vec_resolved: Vec<(u32, Arc<zyron_search::vector::VectorIndex>)> =
        if index_snap.vector.is_empty() {
            Vec::new()
        } else {
            index_snap
                .vector
                .iter()
                .filter_map(|id| ctx.get_vector_index(id.0).map(|idx| (id.0, idx)))
                .collect()
        };
    if fts_resolved.is_empty() && vec_resolved.is_empty() && index_snap.spatial.is_empty() {
        return Ok(());
    }
    let Some(reg) = &ctx.doc_registry else {
        return Err(ZyronError::Internal(
            "search index maintenance requires the document registry".into(),
        ));
    };

    // The ordinal each input row landed at, inverting the write order
    let mut ordinal_of = vec![0u64; order.len()];
    for (ordinal, input_row) in order.iter().enumerate() {
        ordinal_of[*input_row] = ordinal as u64;
    }

    let analyzer = zyron_search::SimpleAnalyzer;
    let mut fts_buf = zyron_search::AnalysisBuffer::new();
    let mut text_buf = String::with_capacity(256);
    let mut input_row = 0usize;
    for batch in batches {
        for row in 0..batch.num_rows {
            let Some(ordinal) = ordinal_of.get(input_row).copied() else {
                break;
            };
            input_row += 1;
            let doc_id = reg.allocate(
                table_entry.id.0,
                zyron_common::RowLocator::Lake {
                    file_id: partition_id,
                    ordinal,
                },
            );

            if !fts_resolved.is_empty() {
                text_buf.clear();
                extract_fts_text_into(batch, row, &table_entry.columns, &mut text_buf);
                for (idx_id, fts_idx) in &fts_resolved {
                    if let Err(e) =
                        fts_idx.add_document_with_buf(doc_id, &text_buf, &analyzer, &mut fts_buf)
                    {
                        return Err(ZyronError::ExecutionError(format!(
                            "full text index {} insert failed on a lake row: {e}",
                            idx_id.0
                        )));
                    }
                }
            }

            for (idx_id, vec_idx) in &vec_resolved {
                let col_id = vec_idx.column_id();
                let Some(vec_bytes) =
                    extract_vector_bytes(batch, row, &table_entry.columns, col_id)
                else {
                    continue;
                };
                let vec_data = bytes_to_f32_slice(vec_bytes);
                if let Err(e) =
                    zyron_search::vector::VectorSearch::insert(vec_idx.as_ref(), doc_id, vec_data)
                {
                    return Err(ZyronError::ExecutionError(format!(
                        "vector index {} insert failed on a lake row: {e}",
                        idx_id
                    )));
                }
            }

            if !index_snap.spatial.is_empty()
                && let Some(spatial_mgr) = &ctx.spatial_manager
            {
                for (idx_id, col_id) in &index_snap.spatial {
                    let Some(tree) = spatial_mgr.get(idx_id.0) else {
                        continue;
                    };
                    let Some(geom_bytes) =
                        extract_column_bytes(batch, row, &table_entry.columns, *col_id)
                    else {
                        continue;
                    };
                    let Ok(geom) = zyron_types::geospatial::decode_wkb(geom_bytes) else {
                        continue;
                    };
                    let mbr = zyron_types::spatial_index::mbr_from_geometry(&geom, tree.dims());
                    tree.insert(zyron_types::spatial_index::LeafEntry {
                        mbr,
                        data: doc_id,
                        deleted: false,
                    });
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DeleteOperator
// ---------------------------------------------------------------------------

/// Pulls rows with tuple IDs from a child scan, logs deletions to WAL,
/// deletes from the heap, and returns the row count.
pub struct DeleteOperator {
    child: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    finished: bool,
}

impl DeleteOperator {
    pub fn new(
        child: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
    ) -> Self {
        Self {
            child,
            ctx,
            table_id,
            finished: false,
        }
    }
}

impl Operator for DeleteOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;

            self.ctx.ensure_writable("DELETE")?;
            self.ctx.ensure_heap_branch_resolved(
                "DELETE",
                &self.ctx.get_table_entry(self.table_id)?.name,
            )?;

            let heap_file = self.ctx.get_heap_file(self.table_id).await?;
            let mut total_deleted: i64 = 0;
            let txn_id = self.ctx.txn_id;

            loop {
                self.ctx.check_cancelled()?;
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };

                // Take exclusive row locks before any mutation so a held
                // FOR UPDATE/SHARE lock blocks this delete and two writers
                // of the same row conflict deterministically instead of
                // both stamping. Keys on RowLocator, so heap and columnar
                // resident rows lock uniformly. Released at commit/abort
                if let Some(locks) = &self.ctx.row_locks
                    && let Some(locs) = exec_batch.locators.as_ref()
                {
                    for loc in locs {
                        locks.lock_row(
                            txn_id as u64,
                            self.table_id.0,
                            *loc,
                            zyron_storage::LockMode::Exclusive,
                        )?;
                    }
                }

                // Exhaustive tier gate: a new RowLocator variant fails to
                // compile here until the delete path handles it
                match exec_batch.tier() {
                    None => {
                        return Err(ZyronError::Internal(
                            "DeleteOperator requires row locators from scan".to_string(),
                        ));
                    }
                    Some(crate::operator::BatchTier::Lake) => {
                        return Err(ZyronError::Internal(
                            "lake resident rows have no DELETE path".to_string(),
                        ));
                    }
                    Some(crate::operator::BatchTier::Columnar)
                    | Some(crate::operator::BatchTier::Heap) => {}
                }

                // Columnar-resident rows: append a supersede to the patch
                // log. No heap delete, no .zyr rewrite. WAL-logged first so
                // the delete and the supersede commit together.
                if let Some(locs) = exec_batch.columnar_pairs() {
                    // A branch delete supersedes on the branch overlay, the
                    // main line keeps the row, mirroring heap page COW at
                    // row granularity
                    let branch = self.ctx.active_branch_id.unwrap_or(0);
                    let te = self.ctx.get_table_entry(self.table_id)?;

                    // Same subsystem sequence as the heap path below. These
                    // operate on the row values the scan materialized, so a
                    // folded row behaves exactly like a heap row.

                    // Apply the ON DELETE referential actions that do not need
                    // the parent's final state, before removing its rows.
                    crate::operator::fk::enforce_parent_delete(
                        &self.ctx,
                        &te,
                        &exec_batch.batch,
                        crate::operator::fk::FkPhase::BeforeWrite,
                    )
                    .await?;

                    // Fire BEFORE DELETE triggers if present.
                    if let Some(ref hook) = self.ctx.dml_hook {
                        let old_tuples = batch_to_tuples(&exec_batch.batch, &te.columns, txn_id);
                        let refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                        if !hook.before_delete(self.table_id.0, &refs, txn_id)? {
                            continue; // Trigger cancelled the delete
                        }
                    }

                    // Fire BEFORE DELETE row/statement triggers in the same txn.
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_BEFORE,
                        zyron_catalog::TriggerEntry::EVENT_DELETE,
                        &exec_batch.batch,
                        &te.columns,
                    )
                    .await?;

                    // Capture old tuples for the CDC hook before the rows are
                    // superseded.
                    let old_tuples_for_cdc = if self.ctx.cdc_hook.is_some() {
                        Some(batch_to_tuples(&exec_batch.batch, &te.columns, txn_id))
                    } else {
                        None
                    };

                    let store = columnar_patch_store(&te)?;
                    let mut last_lsn: u64 = 0;
                    for &(file_id, rowid) in &locs {
                        last_lsn = store.supersede_logged(
                            &self.ctx.wal,
                            self.table_id.0 as u64,
                            branch,
                            file_id,
                            rowid,
                            txn_id as u64,
                        )?;
                        self.ctx.mark_wrote_wal();
                        if self.ctx.recording_undo() {
                            self.ctx.record_columnar_supersede_undo(
                                self.table_id.0,
                                branch,
                                file_id,
                                rowid,
                            );
                        }
                    }
                    total_deleted += locs.len() as i64;

                    // Retire each deleted row's document and remove it from
                    // every search index, keyed by the columnar locator. A row
                    // with no live document has nothing to remove
                    let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
                    let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
                    let spatial_indexes = self.ctx.spatial_indexes_for_table(self.table_id.0);
                    if !fts_indexes.is_empty()
                        || !vec_index_ids.is_empty()
                        || !spatial_indexes.is_empty()
                    {
                        let Some(reg) = &self.ctx.doc_registry else {
                            return Err(ZyronError::Internal(
                                "search index maintenance requires the document registry".into(),
                            ));
                        };
                        for &(file_id, rowid) in &locs {
                            let loc = zyron_common::RowLocator::Columnar {
                                file_id,
                                sys_rowid: rowid,
                            };
                            let Some(doc_id) = reg.take(self.table_id.0, loc) else {
                                continue;
                            };
                            for (idx_id, fts_idx) in &fts_indexes {
                                if let Err(e) = fts_idx.delete_document(doc_id) {
                                    eprintln!("FTS index {} delete failed: {e}", idx_id.0);
                                }
                            }
                            for &idx_id in &vec_index_ids {
                                if let Some(vec_idx) = self.ctx.get_vector_index(idx_id) {
                                    if let Err(e) = zyron_search::vector::VectorSearch::delete(
                                        vec_idx.as_ref(),
                                        doc_id,
                                    ) {
                                        eprintln!("vector index {} delete failed: {e}", idx_id);
                                    }
                                }
                            }
                            if let Some(ref spatial_mgr) = self.ctx.spatial_manager {
                                for (idx_id, _col_id) in &spatial_indexes {
                                    if let Some(tree) = spatial_mgr.get(*idx_id) {
                                        let _ = tree.delete_by_data(&doc_id);
                                    }
                                }
                            }
                        }
                    }

                    if let Some(gm) = &self.ctx.graph_manager {
                        gm.invalidate_for_table(self.table_id.0);
                    }

                    // ON DELETE SET DEFAULT leaves the child referencing the
                    // default key, so it is checked against the parent with
                    // these rows already gone.
                    crate::operator::fk::enforce_parent_delete(
                        &self.ctx,
                        &te,
                        &exec_batch.batch,
                        crate::operator::fk::FkPhase::AfterWrite,
                    )
                    .await?;

                    // Notify CDC hook if present.
                    if let Some(ref hook) = self.ctx.cdc_hook {
                        if let Some(ref old_tuples) = old_tuples_for_cdc {
                            let refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_micros() as i64;
                            if let Err(e) =
                                hook.on_delete(self.table_id.0, &refs, last_lsn, now, txn_id, true)
                            {
                                eprintln!("CDC delete hook failed: {e}");
                            }
                        }
                    }

                    // Fire AFTER DELETE row/statement triggers in the same txn.
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_AFTER,
                        zyron_catalog::TriggerEntry::EVENT_DELETE,
                        &exec_batch.batch,
                        &te.columns,
                    )
                    .await?;
                    continue;
                }

                let tuple_ids = exec_batch.heap_ids().ok_or_else(|| {
                    ZyronError::Internal("DeleteOperator requires row locators from scan".into())
                })?;

                // Branch deletes copy the target page into the branch overlay
                // and tombstone it there, leaving the main line untouched.
                if let Some(bid) = self.ctx.active_branch_id {
                    let deleted = crate::operator::branch_write::branch_delete(
                        &self.ctx,
                        bid,
                        heap_file.heap_file_id(),
                        &tuple_ids,
                    )
                    .await?;
                    total_deleted += deleted as i64;
                    continue;
                }

                // Apply the ON DELETE referential actions that do not need the
                // parent's final state, before removing its rows. Restrict
                // aborts here, cascade and set-null mutate the children first.
                {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    crate::operator::fk::enforce_parent_delete(
                        &self.ctx,
                        &table_entry,
                        &exec_batch.batch,
                        crate::operator::fk::FkPhase::BeforeWrite,
                    )
                    .await?;
                }

                // Fire BEFORE DELETE triggers if present.
                if let Some(ref hook) = self.ctx.dml_hook {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    let old_tuples =
                        batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                    let refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                    if !hook.before_delete(self.table_id.0, &refs, txn_id)? {
                        continue; // Trigger cancelled the delete
                    }
                }

                // Fire BEFORE DELETE row/statement triggers in the same txn.
                {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_BEFORE,
                        zyron_catalog::TriggerEntry::EVENT_DELETE,
                        &exec_batch.batch,
                        &table_entry.columns,
                    )
                    .await?;
                }

                // Capture old tuples for CDC hook (batch data is from the scan).
                let old_tuples_for_cdc = if self.ctx.cdc_hook.is_some() {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    Some(batch_to_tuples(
                        &exec_batch.batch,
                        &table_entry.columns,
                        txn_id,
                    ))
                } else {
                    None
                };

                // Batch WAL log: one CAS + commit for all deletes in this batch.
                let payloads: Vec<Vec<u8>> = tuple_ids.iter().map(tuple_id_payload).collect();
                let batch_records: Vec<(u32, &[u8])> =
                    payloads.iter().map(|p| (txn_id, p.as_slice())).collect();
                let lsns = self.ctx.wal.log_delete_batch(&batch_records)?;
                self.ctx.mark_wrote_wal();
                let last_lsn = lsns.last().copied().unwrap_or(zyron_wal::Lsn::INVALID);

                // MVCC delete: stamp xmax = this txn on each row instead of
                // freeing the slot. Snapshot visibility hides the row once this
                // txn commits; an aborted delete leaves it visible, and vacuum
                // reclaims the space later. B+tree entries are intentionally
                // kept: an index scan rechecks visibility and the key on fetch,
                // so a stale entry can neither resurrect a deleted row nor
                // mismatch a vacuumed-and-reused slot.
                let retain_history = self
                    .ctx
                    .get_table_entry(self.table_id)
                    .map(|t| t.time_travel_retention_secs != 0)
                    .unwrap_or(false);
                let deleted = heap_file
                    .mark_deleted_batch(
                        &tuple_ids,
                        txn_id,
                        self.ctx.snapshot.prune_horizon(),
                        Some(self.ctx.snapshot.status_map().as_ref()),
                        retain_history,
                    )
                    .await?;

                if let Some(gm) = &self.ctx.graph_manager {
                    gm.invalidate_for_table(self.table_id.0);
                }

                // ON DELETE SET DEFAULT leaves the child referencing the
                // default key, so it is checked against the parent with these
                // rows already gone.
                {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    crate::operator::fk::enforce_parent_delete(
                        &self.ctx,
                        &table_entry,
                        &exec_batch.batch,
                        crate::operator::fk::FkPhase::AfterWrite,
                    )
                    .await?;
                }

                // Stamp dirty pages with WAL LSN for checkpoint ordering.
                // Duplicate page_ids are harmless: set_dirty_lsn uses CAS from 0.
                for tid in &tuple_ids {
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(tid.page_id, last_lsn.0);
                }

                total_deleted += deleted as i64;

                // Record a reverse-delete per row so ROLLBACK TO SAVEPOINT
                // clears the xmax stamp and restores the row. Gated on an open
                // savepoint. Only the rows actually stamped are reversible.
                if self.ctx.recording_undo() {
                    let te = self.ctx.get_table_entry(self.table_id)?;
                    for tid in &tuple_ids {
                        self.ctx
                            .record_delete_undo(te.heap_file_id, te.fsm_file_id, *tid);
                    }
                }

                // Retire each deleted row's document and remove it from every
                // search index. A row with no live document (indexed never or
                // index created after insert) has nothing to remove. The
                // ordinal is never reused, so a reused heap slot cannot alias
                // an old document.
                let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
                let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
                let spatial_indexes = self.ctx.spatial_indexes_for_table(self.table_id.0);
                if !fts_indexes.is_empty()
                    || !vec_index_ids.is_empty()
                    || !spatial_indexes.is_empty()
                {
                    let Some(reg) = &self.ctx.doc_registry else {
                        return Err(ZyronError::Internal(
                            "search index maintenance requires the document registry".into(),
                        ));
                    };
                    for tid in &tuple_ids {
                        let Some(doc_id) = reg.take(self.table_id.0, tid.locator()) else {
                            continue;
                        };
                        for (idx_id, fts_idx) in &fts_indexes {
                            if let Err(e) = fts_idx.delete_document(doc_id) {
                                eprintln!("FTS index {} delete failed: {e}", idx_id.0);
                            }
                        }
                        for &idx_id in &vec_index_ids {
                            if let Some(vec_idx) = self.ctx.get_vector_index(idx_id) {
                                if let Err(e) = zyron_search::vector::VectorSearch::delete(
                                    vec_idx.as_ref(),
                                    doc_id,
                                ) {
                                    eprintln!("vector index {} delete failed: {e}", idx_id);
                                }
                            }
                        }
                        if let Some(ref spatial_mgr) = self.ctx.spatial_manager {
                            for (idx_id, _col_id) in &spatial_indexes {
                                if let Some(tree) = spatial_mgr.get(*idx_id) {
                                    let _ = tree.delete_by_data(&doc_id);
                                }
                            }
                        }
                    }
                }

                // Notify CDC hook if present.
                if let Some(ref hook) = self.ctx.cdc_hook {
                    if let Some(ref old_tuples) = old_tuples_for_cdc {
                        let refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as i64;
                        if let Err(e) =
                            hook.on_delete(self.table_id.0, &refs, last_lsn.0, now, txn_id, true)
                        {
                            eprintln!("CDC delete hook failed: {e}");
                        }
                    }
                }

                // Fire AFTER DELETE row/statement triggers in the same txn.
                {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_AFTER,
                        zyron_catalog::TriggerEntry::EVENT_DELETE,
                        &exec_batch.batch,
                        &table_entry.columns,
                    )
                    .await?;
                }
            }

            if let Some(stats) = self.ctx.table_io_stats_for(self.table_id.0) {
                stats.record_deletes(total_deleted.max(0) as u64);
            }

            Ok(Some(ExecutionBatch::new(count_batch(total_deleted))))
        })
    }
}

// ---------------------------------------------------------------------------
// UpdateOperator
// ---------------------------------------------------------------------------

/// Pulls rows with tuple IDs from a child scan, evaluates assignment
/// expressions to produce updated column values, deletes old tuples,
/// inserts new tuples, and returns the row count.
pub struct UpdateOperator {
    child: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    assignments: Vec<BoundAssignment>,
    input_schema: Vec<LogicalColumn>,
    /// CHECK constraint predicates (bound at table_idx 0) enforced on the
    /// updated row image before it is written.
    check_constraints: Vec<zyron_planner::binder::BoundExpr>,
    /// Set when any assignment value holds a correlated subquery, evaluates
    /// every assignment value per batch with per row subquery execution.
    correlated_values: Option<crate::correlated::CorrelatedValues>,
    /// Parameter values the assignments read instead of the context's. The
    /// foreign-key cascade sets this: it writes a parent key it read out of
    /// a batch, and a UUID, a sixteen-byte integer or a binary key has no
    /// literal that carries it exactly, while a parameter carries the
    /// scalar itself.
    params: Option<Vec<ScalarValue>>,
    finished: bool,
}

impl UpdateOperator {
    pub fn new(
        child: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        assignments: Vec<BoundAssignment>,
        input_schema: Vec<LogicalColumn>,
        check_constraints: Vec<zyron_planner::binder::BoundExpr>,
    ) -> Self {
        Self {
            child,
            ctx,
            table_id,
            assignments,
            input_schema,
            check_constraints,
            correlated_values: None,
            params: None,
            finished: false,
        }
    }

    /// Supplies the parameter values assignment and CHECK evaluation read,
    /// for a write the executor drives itself rather than one a statement
    /// bound.
    pub fn with_params(mut self, params: Vec<ScalarValue>) -> Self {
        self.params = Some(params);
        self
    }

    #[inline]
    fn params(&self) -> &[ScalarValue] {
        match &self.params {
            Some(params) => params,
            None => &self.ctx.params,
        }
    }

    /// Routes assignment evaluation through per row correlated subquery
    /// execution, used when a SET value references the updated table
    pub fn with_correlated_values(mut self, cv: crate::correlated::CorrelatedValues) -> Self {
        self.correlated_values = Some(cv);
        self
    }
}

impl Operator for UpdateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;

            self.ctx.ensure_writable("UPDATE")?;

            let table_entry = self.ctx.get_table_entry(self.table_id)?;
            self.ctx
                .ensure_heap_branch_resolved("UPDATE", &table_entry.name)?;
            let heap_file = self.ctx.get_heap_file(self.table_id).await?;
            let mut total_updated: i64 = 0;
            let txn_id = self.ctx.txn_id;

            // Search indexes have no MVCC visibility recheck, so their
            // maintenance is deferred into these buffers and applied only after
            // the whole statement's batch loop completes. A mid-statement error
            // returns early and drops the buffers, leaving no orphaned entries.
            // Old and new documents carry distinct registry ordinals, so the
            // apply order cannot alias.
            let mut deferred_fts_deletes: Vec<u64> = Vec::new();
            let mut deferred_fts_inserts: Vec<(u64, String)> = Vec::new();
            let mut deferred_vec_deletes: Vec<u64> = Vec::new();
            let mut deferred_vec_inserts: Vec<(u32, u64, Vec<f32>)> = Vec::new();
            let mut deferred_spatial_deletes: Vec<u64> = Vec::new();
            let mut deferred_spatial_inserts: Vec<(u32, zyron_types::spatial_index::Mbr, u64)> =
                Vec::new();

            loop {
                self.ctx.check_cancelled()?;
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };

                // Take exclusive row locks before any mutation so a held
                // FOR UPDATE/SHARE lock blocks this update and two writers
                // of the same row conflict deterministically instead of
                // both stamping. Keys on RowLocator, so heap and columnar
                // resident rows lock uniformly. Released at commit/abort
                if let Some(locks) = &self.ctx.row_locks
                    && let Some(locs) = exec_batch.locators.as_ref()
                {
                    for loc in locs {
                        locks.lock_row(
                            txn_id as u64,
                            self.table_id.0,
                            *loc,
                            zyron_storage::LockMode::Exclusive,
                        )?;
                    }
                }

                // Exhaustive tier gate: a new RowLocator variant fails to
                // compile here until the update path handles it
                match exec_batch.tier() {
                    None => {
                        return Err(ZyronError::Internal(
                            "UpdateOperator requires row locators from scan".to_string(),
                        ));
                    }
                    Some(crate::operator::BatchTier::Lake) => {
                        return Err(ZyronError::Internal(
                            "lake resident rows have no UPDATE path".to_string(),
                        ));
                    }
                    Some(crate::operator::BatchTier::Columnar)
                    | Some(crate::operator::BatchTier::Heap) => {}
                }

                // Columnar-resident rows: write one epoch-tagged value patch
                // per assigned column to the patch log. The old columnar
                // value remains the version for snapshots that predate this
                // transaction; this patch is the version for later ones. No
                // heap round trip, no .zyr rewrite.
                if let Some(locs) = exec_batch.columnar_pairs() {
                    // A branch update patches the branch overlay, the main
                    // line keeps its values
                    let branch = self.ctx.active_branch_id.unwrap_or(0);
                    let store = columnar_patch_store(&table_entry)?;

                    // Evaluate every assignment, cast to the column type, and
                    // build the merged row image (scanned columns with assigned
                    // ones replaced). The image is what CHECK constraints are
                    // evaluated against before any patch is written, so a
                    // violating columnar update aborts with no effect, matching
                    // the heap path.
                    let mut updated_columns = exec_batch.batch.columns.clone();
                    let mut patches: Vec<(u32, TypeId, usize, crate::column::Column)> =
                        Vec::with_capacity(self.assignments.len());
                    let precomputed = match &self.correlated_values {
                        Some(cv) => Some(cv.eval(&self.ctx, &exec_batch.batch).await?),
                        None => None,
                    };
                    for (assign_idx, assignment) in self.assignments.iter().enumerate() {
                        let new_col = match &precomputed {
                            Some(cols) => cols[assign_idx].clone(),
                            None => evaluate(
                                &assignment.value,
                                &exec_batch.batch,
                                &self.input_schema,
                                self.params(),
                            )?,
                        };
                        let ce = table_entry
                            .columns
                            .iter()
                            .find(|c| c.id == assignment.column_id)
                            .ok_or_else(|| {
                                ZyronError::Internal(format!(
                                    "assignment column {:?} not in table",
                                    assignment.column_id
                                ))
                            })?;
                        let new_col = if new_col.type_id != ce.type_id {
                            crate::compute::cast_column(&new_col, ce.type_id)?
                        } else {
                            new_col
                        };
                        if let Some(idx) = self
                            .input_schema
                            .iter()
                            .position(|lc| lc.column_id == assignment.column_id)
                        {
                            updated_columns[idx] = new_col.clone();
                        }
                        let phys = ce.physical_type_id();
                        patches.push((
                            ce.id.0 as u32,
                            phys,
                            phys.fixed_size().unwrap_or(0),
                            new_col,
                        ));
                    }

                    let mut updated_batch = DataBatch::new(updated_columns);
                    normalize_array_elements(&mut updated_batch, &table_entry.columns)?;
                    enforce_check_constraints(
                        &self.check_constraints,
                        &updated_batch,
                        &table_entry.columns,
                        self.params(),
                    )?;

                    // Same subsystem sequence as the heap path below, all of
                    // it value based so a folded row behaves exactly like a
                    // heap row.

                    // Enforce child side foreign keys on the post update image
                    // before any write.
                    crate::operator::fk::check_child_fks(&self.ctx, &table_entry, &updated_batch)
                        .await?
                        .deny_diversion(&table_entry.name)?;

                    // Enforce unique constraints on the post update image.
                    // The rows being updated are excluded by their columnar
                    // locators so a row never conflicts with its own entries.
                    {
                        let index_snap = self.ctx.index_snapshot_for_table(self.table_id.0);
                        let exclude_locators: Vec<zyron_common::RowLocator> = locs
                            .iter()
                            .map(|&(file_id, rowid)| zyron_common::RowLocator::Columnar {
                                file_id,
                                sys_rowid: rowid,
                            })
                            .collect();
                        check_unique_constraints(
                            &self.ctx,
                            &table_entry,
                            &updated_batch,
                            &index_snap,
                            &exclude_locators,
                        )
                        .await?;
                    }

                    // Apply the ON UPDATE referential actions that do not need
                    // the new key to exist, so a rejection lands before any
                    // row is written.
                    crate::operator::fk::enforce_parent_update(
                        &self.ctx,
                        &table_entry,
                        &exec_batch.batch,
                        &updated_batch,
                        crate::operator::fk::FkPhase::BeforeWrite,
                    )
                    .await?;

                    // Fire BEFORE UPDATE triggers if present.
                    if let Some(ref hook) = self.ctx.dml_hook {
                        let old_tuples =
                            batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                        let new_tuples =
                            batch_to_tuples(&updated_batch, &table_entry.columns, txn_id);
                        let old_refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                        let new_refs: Vec<&[u8]> = new_tuples.iter().map(|t| t.data()).collect();
                        if !hook.before_update(self.table_id.0, &old_refs, &new_refs, txn_id)? {
                            continue; // Trigger cancelled the update
                        }
                    }

                    // Fire BEFORE UPDATE row/statement triggers (NEW image).
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_BEFORE,
                        zyron_catalog::TriggerEntry::EVENT_UPDATE,
                        &updated_batch,
                        &table_entry.columns,
                    )
                    .await?;

                    let mut last_lsn: u64 = 0;
                    for (col_id, phys, vsize, new_col) in &patches {
                        for (r, &(file_id, rowid)) in locs.iter().enumerate() {
                            let sv = new_col.data.get_scalar(r);
                            let bytes = encode_scalar_value(*phys, &sv, *vsize);
                            last_lsn = store.patch_logged(
                                &self.ctx.wal,
                                self.table_id.0 as u64,
                                branch,
                                file_id,
                                rowid,
                                *col_id,
                                txn_id as u64,
                                &bytes,
                            )?;
                            self.ctx.mark_wrote_wal();
                            if self.ctx.recording_undo() {
                                self.ctx.record_columnar_patch_undo(
                                    self.table_id.0,
                                    branch,
                                    file_id,
                                    rowid,
                                    *col_id,
                                );
                            }
                        }
                    }
                    total_updated += locs.len() as i64;

                    // Maintain B+tree indexes: add the new image's keys under
                    // the rows' columnar locators. Old-value entries are
                    // intentionally kept, the probes recheck overlay liveness
                    // and the current value on fetch, mirroring the heap
                    // MVCC-retention contract
                    {
                        let index_snap = self.ctx.index_snapshot_for_table(self.table_id.0);
                        let update_locators: Vec<zyron_common::RowLocator> = locs
                            .iter()
                            .map(|&(file_id, rowid)| zyron_common::RowLocator::Columnar {
                                file_id,
                                sys_rowid: rowid,
                            })
                            .collect();
                        maintain_btree_insert(
                            &self.ctx,
                            &table_entry,
                            &updated_batch,
                            &update_locators,
                            &index_snap,
                        );
                    }

                    // Queue search index maintenance keyed by the columnar
                    // locator: the old row version's document retires and the
                    // patched version gets a fresh ordinal at the same locator,
                    // buffered and applied after the statement succeeds
                    let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
                    let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
                    let spatial_indexes = self.ctx.spatial_indexes_for_table(self.table_id.0);
                    if !fts_indexes.is_empty()
                        || !vec_index_ids.is_empty()
                        || !spatial_indexes.is_empty()
                    {
                        let Some(reg) = &self.ctx.doc_registry else {
                            return Err(ZyronError::Internal(
                                "search index maintenance requires the document registry".into(),
                            ));
                        };
                        let mut text_buf = String::with_capacity(256);
                        for (row_idx, &(file_id, rowid)) in locs.iter().enumerate() {
                            let loc = zyron_common::RowLocator::Columnar {
                                file_id,
                                sys_rowid: rowid,
                            };
                            if let Some(doc_id) = reg.take(self.table_id.0, loc) {
                                if !fts_indexes.is_empty() {
                                    deferred_fts_deletes.push(doc_id);
                                }
                                if !vec_index_ids.is_empty() {
                                    deferred_vec_deletes.push(doc_id);
                                }
                                if !spatial_indexes.is_empty() {
                                    deferred_spatial_deletes.push(doc_id);
                                }
                            }
                            let doc_id = reg.allocate(self.table_id.0, loc);
                            if !fts_indexes.is_empty() {
                                text_buf.clear();
                                extract_fts_text_into(
                                    &updated_batch,
                                    row_idx,
                                    &table_entry.columns,
                                    &mut text_buf,
                                );
                                deferred_fts_inserts.push((doc_id, text_buf.clone()));
                            }
                            for &idx_id in &vec_index_ids {
                                let Some(vec_idx) = self.ctx.get_vector_index(idx_id) else {
                                    continue;
                                };
                                let col_id = vec_idx.column_id();
                                if let Some(vec_bytes) = extract_vector_bytes(
                                    &updated_batch,
                                    row_idx,
                                    &table_entry.columns,
                                    col_id,
                                ) {
                                    let vec_data = bytes_to_f32_slice(vec_bytes).to_vec();
                                    deferred_vec_inserts.push((idx_id, doc_id, vec_data));
                                }
                            }
                            for (idx_id, col_id) in &spatial_indexes {
                                let Some(spatial_mgr) = &self.ctx.spatial_manager else {
                                    continue;
                                };
                                let Some(tree) = spatial_mgr.get(*idx_id) else {
                                    continue;
                                };
                                let Some(geom_bytes) = extract_column_bytes(
                                    &updated_batch,
                                    row_idx,
                                    &table_entry.columns,
                                    *col_id,
                                ) else {
                                    continue;
                                };
                                let Ok(geom) = zyron_types::geospatial::decode_wkb(geom_bytes)
                                else {
                                    continue;
                                };
                                let mbr = zyron_types::spatial_index::mbr_from_geometry(
                                    &geom,
                                    tree.dims(),
                                );
                                deferred_spatial_inserts.push((*idx_id, mbr, doc_id));
                            }
                        }
                    }

                    if let Some(gm) = &self.ctx.graph_manager {
                        gm.invalidate_for_table(self.table_id.0);
                    }

                    // ON UPDATE CASCADE points children at the new key, which
                    // their own foreign keys read, so it runs once the parent
                    // rows carry it.
                    crate::operator::fk::enforce_parent_update(
                        &self.ctx,
                        &table_entry,
                        &exec_batch.batch,
                        &updated_batch,
                        crate::operator::fk::FkPhase::AfterWrite,
                    )
                    .await?;

                    // Notify CDC hook if present.
                    if let Some(ref hook) = self.ctx.cdc_hook {
                        let old_tuples =
                            batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                        let new_tuples =
                            batch_to_tuples(&updated_batch, &table_entry.columns, txn_id);
                        let old_refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                        let new_refs: Vec<&[u8]> = new_tuples.iter().map(|t| t.data()).collect();
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as i64;
                        if let Err(e) = hook.on_update(
                            self.table_id.0,
                            &old_refs,
                            &new_refs,
                            last_lsn,
                            now,
                            txn_id,
                            true,
                        ) {
                            eprintln!("CDC update hook failed: {e}");
                        }
                    }

                    // Fire AFTER UPDATE row/statement triggers (NEW image).
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_AFTER,
                        zyron_catalog::TriggerEntry::EVENT_UPDATE,
                        &updated_batch,
                        &table_entry.columns,
                    )
                    .await?;
                    continue;
                }

                let tuple_ids = exec_batch.heap_ids().ok_or_else(|| {
                    ZyronError::Internal("UpdateOperator requires row locators from scan".into())
                })?;

                // Build the updated batch by cloning original columns
                // and replacing assigned columns with new values.
                let mut updated_columns = exec_batch.batch.columns.clone();

                let precomputed = match &self.correlated_values {
                    Some(cv) => Some(cv.eval(&self.ctx, &exec_batch.batch).await?),
                    None => None,
                };
                for (assign_idx, assignment) in self.assignments.iter().enumerate() {
                    let new_col = match &precomputed {
                        Some(cols) => cols[assign_idx].clone(),
                        None => evaluate(
                            &assignment.value,
                            &exec_batch.batch,
                            &self.input_schema,
                            self.params(),
                        )?,
                    };

                    // Coerce the assigned value to the target column's type. The
                    // binder does not cast assignment expressions, so an integer
                    // literal evaluates to i64; without this the row encoder
                    // zero-fills a narrower column (Int32 from Int64) and the
                    // update silently writes 0.
                    let target_type = table_entry
                        .columns
                        .iter()
                        .find(|c| c.id == assignment.column_id)
                        .map(|c| c.type_id);
                    let new_col = match target_type {
                        Some(t) if new_col.type_id != t => {
                            crate::compute::cast_column(&new_col, t)?
                        }
                        _ => new_col,
                    };

                    // Find the column index matching this assignment's column_id.
                    let col_idx = self
                        .input_schema
                        .iter()
                        .position(|lc| lc.column_id == assignment.column_id)
                        .ok_or_else(|| {
                            ZyronError::Internal(format!(
                                "Assignment column {:?} not found in schema",
                                assignment.column_id
                            ))
                        })?;

                    updated_columns[col_idx] = new_col;
                }

                let mut updated_batch = DataBatch::new(updated_columns);
                normalize_array_elements(&mut updated_batch, &table_entry.columns)?;

                // Enforce CHECK constraints on the updated row image before any
                // write so a violating update aborts with no effect.
                enforce_check_constraints(
                    &self.check_constraints,
                    &updated_batch,
                    &table_entry.columns,
                    self.params(),
                )?;

                // Branch updates tombstone the old image in the branch overlay
                // (copying the source page first) and append the new image to
                // the branch, keeping the main line untouched.
                if let Some(bid) = self.ctx.active_branch_id {
                    let new_tuples = batch_to_tuples(&updated_batch, &table_entry.columns, txn_id);
                    crate::operator::branch_write::branch_delete(
                        &self.ctx,
                        bid,
                        heap_file.heap_file_id(),
                        &tuple_ids,
                    )
                    .await?;
                    crate::operator::branch_write::branch_insert(
                        &self.ctx,
                        bid,
                        heap_file.heap_file_id(),
                        &new_tuples,
                    )
                    .await?;
                    total_updated += tuple_ids.len() as i64;
                    continue;
                }

                // Enforce child-side foreign keys on the post-update image
                // before any write so a violation aborts cleanly.
                crate::operator::fk::check_child_fks(&self.ctx, &table_entry, &updated_batch)
                    .await?
                    .deny_diversion(&table_entry.name)?;

                // Enforce unique constraints on the post-update image before any
                // write, excluding the rows being updated (a row keeping its own
                // value is not a conflict). Abort does not undo heap rows.
                {
                    let index_snap = self.ctx.index_snapshot_for_table(self.table_id.0);
                    #[cfg(feature = "profile")]
                    let _uc_span =
                        zyron_common::profile::scope(zyron_common::profile::Phase::ExecUniqueCheck);
                    let exclude_locators: Vec<zyron_common::RowLocator> =
                        tuple_ids.iter().map(|t| t.locator()).collect();
                    check_unique_constraints(
                        &self.ctx,
                        &table_entry,
                        &updated_batch,
                        &index_snap,
                        &exclude_locators,
                    )
                    .await?;
                }

                // Apply the ON UPDATE referential actions that do not need the
                // new key to exist, so a rejection lands before any row is
                // written.
                crate::operator::fk::enforce_parent_update(
                    &self.ctx,
                    &table_entry,
                    &exec_batch.batch,
                    &updated_batch,
                    crate::operator::fk::FkPhase::BeforeWrite,
                )
                .await?;

                let new_tuples = batch_to_tuples(&updated_batch, &table_entry.columns, txn_id);

                // Fire BEFORE UPDATE triggers if present.
                if let Some(ref hook) = self.ctx.dml_hook {
                    let old_tuples =
                        batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                    let old_refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                    let new_refs: Vec<&[u8]> = new_tuples.iter().map(|t| t.data()).collect();
                    if !hook.before_update(self.table_id.0, &old_refs, &new_refs, txn_id)? {
                        continue; // Trigger cancelled the update
                    }
                }

                // Fire BEFORE UPDATE row/statement triggers (NEW image) in txn.
                crate::trigger::fire_row_triggers(
                    &self.ctx,
                    self.table_id,
                    zyron_catalog::TriggerEntry::TIMING_BEFORE,
                    zyron_catalog::TriggerEntry::EVENT_UPDATE,
                    &updated_batch,
                    &table_entry.columns,
                )
                .await?;

                // Batch WAL log deletes: one CAS + commit for all.
                let delete_payloads: Vec<Vec<u8>> =
                    tuple_ids.iter().map(tuple_id_payload).collect();
                let delete_records: Vec<(u32, &[u8])> = delete_payloads
                    .iter()
                    .map(|p| (txn_id, p.as_slice()))
                    .collect();
                let del_lsns = self.ctx.wal.log_delete_batch(&delete_records)?;
                self.ctx.mark_wrote_wal();
                let del_last_lsn = del_lsns.last().copied().unwrap_or(zyron_wal::Lsn::INVALID);
                // MVCC: stamp xmax on the old image rather than freeing it, so an
                // aborted update leaves the original row visible.
                heap_file
                    .mark_deleted_batch(
                        &tuple_ids,
                        txn_id,
                        self.ctx.snapshot.prune_horizon(),
                        Some(self.ctx.snapshot.status_map().as_ref()),
                        table_entry.time_travel_retention_secs != 0,
                    )
                    .await?;

                if let Some(gm) = &self.ctx.graph_manager {
                    gm.invalidate_for_table(self.table_id.0);
                }

                // Record reverse-delete for each old image so ROLLBACK TO
                // SAVEPOINT restores it (an UPDATE is delete-old then
                // insert-new). Gated on an open savepoint. Recorded before the
                // reverse-insert below so the log reverses insert-then-delete.
                if self.ctx.recording_undo() {
                    for tid in &tuple_ids {
                        self.ctx.record_delete_undo(
                            table_entry.heap_file_id,
                            table_entry.fsm_file_id,
                            *tid,
                        );
                    }
                }

                // Stamp deleted pages with WAL LSN for checkpoint ordering.
                // Duplicate page_ids are harmless: set_dirty_lsn uses CAS from 0.
                for tid in &tuple_ids {
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(tid.page_id, del_last_lsn.0);
                }

                // Batch WAL log inserts: one CAS + commit for all.
                let insert_records: Vec<(u32, &[u8])> =
                    new_tuples.iter().map(|t| (txn_id, t.data())).collect();
                let ins_lsns = self.ctx.wal.log_insert_batch(&insert_records)?;
                let ins_last_lsn = ins_lsns.last().copied().unwrap_or(zyron_wal::Lsn::INVALID);
                #[cfg(feature = "profile")]
                let _heap_span =
                    zyron_common::profile::scope(zyron_common::profile::Phase::ExecHeapInsert);
                let new_tuple_ids = heap_file.insert_batch(&new_tuples).await?;
                #[cfg(feature = "profile")]
                drop(_heap_span);

                // Record reverse-insert for each new image so ROLLBACK TO
                // SAVEPOINT self-deletes it. Gated on an open savepoint.
                if self.ctx.recording_undo() {
                    for tid in &new_tuple_ids {
                        self.ctx.record_insert_undo(
                            table_entry.heap_file_id,
                            table_entry.fsm_file_id,
                            *tid,
                        );
                    }
                }

                // Stamp inserted pages with WAL LSN for checkpoint ordering.
                for tid in &new_tuple_ids {
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(tid.page_id, ins_last_lsn.0);
                }

                total_updated += tuple_ids.len() as i64;

                // Maintain B+tree indexes: add the new image's keys. The old
                // image's entries are intentionally kept (the old row is only
                // xmax-stamped, not removed); an index scan rechecks visibility
                // and the key on fetch, so the old entry resolves to the now
                // invisible old row and is skipped. Composite (value, tid) keys
                // mean the new entry never collides with the old one.
                {
                    let index_snap = self.ctx.index_snapshot_for_table(self.table_id.0);
                    #[cfg(feature = "profile")]
                    let _idx_span =
                        zyron_common::profile::scope(zyron_common::profile::Phase::ExecIndexInsert);
                    let index_locators: Vec<zyron_common::RowLocator> =
                        new_tuple_ids.iter().map(|t| t.locator()).collect();
                    maintain_btree_insert(
                        &self.ctx,
                        &table_entry,
                        &updated_batch,
                        &index_locators,
                        &index_snap,
                    );
                }

                // Queue search index maintenance: the old row version's
                // document retires and the new version gets a fresh ordinal,
                // buffered and applied after the statement succeeds. A row
                // with no live document was never indexed and queues nothing.
                let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
                let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
                let spatial_indexes = self.ctx.spatial_indexes_for_table(self.table_id.0);
                let has_search_indexes = !fts_indexes.is_empty()
                    || !vec_index_ids.is_empty()
                    || !spatial_indexes.is_empty();
                if has_search_indexes {
                    let Some(reg) = &self.ctx.doc_registry else {
                        return Err(ZyronError::Internal(
                            "search index maintenance requires the document registry".into(),
                        ));
                    };
                    for tid in &tuple_ids {
                        if let Some(doc_id) = reg.take(self.table_id.0, tid.locator()) {
                            if !fts_indexes.is_empty() {
                                deferred_fts_deletes.push(doc_id);
                            }
                            if !vec_index_ids.is_empty() {
                                deferred_vec_deletes.push(doc_id);
                            }
                            if !spatial_indexes.is_empty() {
                                deferred_spatial_deletes.push(doc_id);
                            }
                        }
                    }
                    let mut text_buf = String::with_capacity(256);
                    for (row_idx, tid) in new_tuple_ids.iter().enumerate() {
                        let doc_id = reg.allocate(table_entry.id.0, tid.locator());
                        if !fts_indexes.is_empty() {
                            text_buf.clear();
                            extract_fts_text_into(
                                &updated_batch,
                                row_idx,
                                &table_entry.columns,
                                &mut text_buf,
                            );
                            deferred_fts_inserts.push((doc_id, text_buf.clone()));
                        }
                        for &idx_id in &vec_index_ids {
                            let Some(vec_idx) = self.ctx.get_vector_index(idx_id) else {
                                continue;
                            };
                            let col_id = vec_idx.column_id();
                            if let Some(vec_bytes) = extract_vector_bytes(
                                &updated_batch,
                                row_idx,
                                &table_entry.columns,
                                col_id,
                            ) {
                                let vec_data = bytes_to_f32_slice(vec_bytes).to_vec();
                                deferred_vec_inserts.push((idx_id, doc_id, vec_data));
                            }
                        }
                        // spatial entries queue the new geometry's MBR so the
                        // R-tree tracks moved geometries instead of keeping a
                        // dead entry per update
                        for (idx_id, col_id) in &spatial_indexes {
                            let Some(spatial_mgr) = &self.ctx.spatial_manager else {
                                continue;
                            };
                            let Some(tree) = spatial_mgr.get(*idx_id) else {
                                continue;
                            };
                            let Some(geom_bytes) = extract_column_bytes(
                                &updated_batch,
                                row_idx,
                                &table_entry.columns,
                                *col_id,
                            ) else {
                                continue;
                            };
                            let Ok(geom) = zyron_types::geospatial::decode_wkb(geom_bytes) else {
                                continue;
                            };
                            let mbr =
                                zyron_types::spatial_index::mbr_from_geometry(&geom, tree.dims());
                            deferred_spatial_inserts.push((*idx_id, mbr, doc_id));
                        }
                    }
                }

                // ON UPDATE CASCADE points children at the new key, which their
                // own foreign keys read, so it runs once the parent rows carry
                // it.
                crate::operator::fk::enforce_parent_update(
                    &self.ctx,
                    &table_entry,
                    &exec_batch.batch,
                    &updated_batch,
                    crate::operator::fk::FkPhase::AfterWrite,
                )
                .await?;

                // Notify CDC hook if present.
                if let Some(ref hook) = self.ctx.cdc_hook {
                    let old_tuples =
                        batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                    let old_slices: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                    let new_refs_data: Vec<&[u8]> = new_tuples.iter().map(|t| t.data()).collect();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as i64;
                    if let Err(e) = hook.on_update(
                        self.table_id.0,
                        &old_slices,
                        &new_refs_data,
                        ins_last_lsn.0,
                        now,
                        txn_id,
                        true,
                    ) {
                        eprintln!("CDC update hook failed: {e}");
                    }
                }

                // Fire AFTER UPDATE row/statement triggers (NEW image) in txn.
                crate::trigger::fire_row_triggers(
                    &self.ctx,
                    self.table_id,
                    zyron_catalog::TriggerEntry::TIMING_AFTER,
                    zyron_catalog::TriggerEntry::EVENT_UPDATE,
                    &updated_batch,
                    &table_entry.columns,
                )
                .await?;
            }

            // The statement succeeded, so apply the buffered FTS and vector
            // index maintenance now. Deletes run before inserts so a reused
            // heap slot's new entry survives a queued delete of the old one.
            let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
            if !fts_indexes.is_empty() {
                for &doc_id in &deferred_fts_deletes {
                    for (idx_id, fts_idx) in &fts_indexes {
                        if let Err(e) = fts_idx.delete_document(doc_id) {
                            eprintln!("FTS index {} update-delete failed: {e}", idx_id.0);
                        }
                    }
                }
                let analyzer = zyron_search::SimpleAnalyzer;
                let mut fts_buf = zyron_search::AnalysisBuffer::new();
                for (doc_id, text) in &deferred_fts_inserts {
                    for (idx_id, fts_idx) in &fts_indexes {
                        if let Err(e) =
                            fts_idx.add_document_with_buf(*doc_id, text, &analyzer, &mut fts_buf)
                        {
                            eprintln!("FTS index {} update-insert failed: {e}", idx_id.0);
                        }
                    }
                }
            }

            let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
            if !vec_index_ids.is_empty() {
                for &vec_id in &deferred_vec_deletes {
                    for &idx_id in &vec_index_ids {
                        if let Some(vec_idx) = self.ctx.get_vector_index(idx_id) {
                            if let Err(e) =
                                zyron_search::vector::VectorSearch::delete(vec_idx.as_ref(), vec_id)
                            {
                                eprintln!("vector index {idx_id} update-delete failed: {e}");
                            }
                        }
                    }
                }
                for (idx_id, vec_id, vec_data) in &deferred_vec_inserts {
                    if let Some(vec_idx) = self.ctx.get_vector_index(*idx_id) {
                        if let Err(e) = zyron_search::vector::VectorSearch::insert(
                            vec_idx.as_ref(),
                            *vec_id,
                            vec_data,
                        ) {
                            eprintln!("vector index {idx_id} update-insert failed: {e}");
                        }
                    }
                }
            }

            if let Some(ref spatial_mgr) = self.ctx.spatial_manager {
                let spatial_indexes = self.ctx.spatial_indexes_for_table(self.table_id.0);
                for &doc_id in &deferred_spatial_deletes {
                    for (idx_id, _col_id) in &spatial_indexes {
                        if let Some(tree) = spatial_mgr.get(*idx_id) {
                            let _ = tree.delete_by_data(&doc_id);
                        }
                    }
                }
                for (idx_id, mbr, doc_id) in &deferred_spatial_inserts {
                    if let Some(tree) = spatial_mgr.get(*idx_id) {
                        tree.insert(zyron_types::spatial_index::LeafEntry {
                            mbr: *mbr,
                            data: *doc_id,
                            deleted: false,
                        });
                    }
                }
            }

            if let Some(stats) = self.ctx.table_io_stats_for(self.table_id.0) {
                stats.record_updates(total_updated.max(0) as u64);
            }

            Ok(Some(ExecutionBatch::new(count_batch(total_updated))))
        })
    }
}

#[cfg(test)]
mod b6_key_tests {
    use super::*;
    use crate::column::{Column, ColumnData};
    use zyron_common::TypeId;

    #[test]
    fn test_cell_key_encoding_matches_batch_key_encoding() {
        // every (column data, physical type, raw cell bytes) triple must key
        // identically through both encoders or fold re-keys silently miss
        let cases: Vec<(ColumnData, TypeId, Vec<u8>)> = vec![
            (ColumnData::Int8(vec![-5]), TypeId::Int8, vec![(-5i8) as u8]),
            (
                ColumnData::Int16(vec![-300]),
                TypeId::Int16,
                (-300i16).to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Int32(vec![-70_000]),
                TypeId::Int32,
                (-70_000i32).to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Int64(vec![-1]),
                TypeId::Int64,
                (-1i64).to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Int64(vec![7_777_777]),
                TypeId::Int64,
                7_777_777i64.to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Int128(vec![-1_000_000_000_000]),
                TypeId::Int128,
                (-1_000_000_000_000i128).to_le_bytes().to_vec(),
            ),
            (ColumnData::UInt8(vec![200]), TypeId::UInt8, vec![200u8]),
            (
                ColumnData::UInt16(vec![50_000]),
                TypeId::UInt16,
                50_000u16.to_le_bytes().to_vec(),
            ),
            (
                ColumnData::UInt32(vec![4_000_000_000]),
                TypeId::UInt32,
                4_000_000_000u32.to_le_bytes().to_vec(),
            ),
            (
                ColumnData::UInt64(vec![u64::MAX]),
                TypeId::UInt64,
                u64::MAX.to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Float64(vec![-3.5]),
                TypeId::Float64,
                (-3.5f64).to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Float64(vec![2.25]),
                TypeId::Float64,
                2.25f64.to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Float32(vec![-1.5]),
                TypeId::Float32,
                (-1.5f32).to_le_bytes().to_vec(),
            ),
            (
                ColumnData::Utf8(vec!["hello".to_string()]),
                TypeId::Text,
                b"hello".to_vec(),
            ),
            (
                ColumnData::Binary(vec![vec![1, 2, 3]]),
                TypeId::Binary,
                vec![1, 2, 3],
            ),
        ];
        for (data, type_id, cell) in cases {
            let col = Column::new(data, type_id);
            let batch = DataBatch::new(vec![col]);
            let mut from_batch = Vec::new();
            assert!(
                encode_btree_key_into(&batch, 0, 0, type_id, &mut from_batch),
                "batch encoder rejected {type_id:?}"
            );
            let mut from_cell = Vec::new();
            assert!(
                encode_btree_key_from_cell(type_id, &cell, &mut from_cell),
                "cell encoder rejected {type_id:?}"
            );
            assert_eq!(from_batch, from_cell, "key drift for {type_id:?}");
        }
    }

    #[test]
    fn test_i128_index_key_order_preserving_with_negatives() {
        // i128 values incl. pre-1970 (negative) picosecond timestamps must
        // produce big-endian keys that sort in numeric order.
        let vals: Vec<i128> = vec![
            i128::MIN / 2,
            -1_000_000_000_000,
            -1,
            0,
            1,
            1_700_000_000_000_000_000_000,
            i128::MAX / 2,
        ];
        let col = Column::new_ts(
            ColumnData::Int128(vals.clone()),
            TypeId::TimestampTz,
            Some(9),
        );
        let batch = DataBatch::new(vec![col]);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for r in 0..vals.len() {
            let mut buf = Vec::new();
            assert!(encode_btree_key_into(
                &batch,
                r,
                0,
                TypeId::TimestampTz,
                &mut buf
            ));
            assert_eq!(buf.len(), 16);
            keys.push(buf);
        }
        // Byte-lexicographic order of keys must match numeric order of values.
        for i in 1..keys.len() {
            assert!(
                keys[i - 1] < keys[i],
                "key order broken at {i}: {:?} vs {:?}",
                vals[i - 1],
                vals[i]
            );
        }
    }
}

//! Foreign-key referential integrity enforcement for DML operators.
//!
//! Child-side: INSERT and UPDATE verify that each non-null foreign-key value
//! has a matching live row in the referenced parent table (MATCH SIMPLE, so a
//! row with any null FK column is skipped). Parent-side: DELETE and UPDATE of a
//! referenced key apply the constraint's ON DELETE / ON UPDATE action to the
//! referencing child rows.
//!
//! Existence checks use the parent's B+tree index on the referenced column for
//! an O(1) probe when one exists, and fall back to a one-pass key-set build of
//! the parent table otherwise. All reads honor the statement's MVCC snapshot.

use std::cell::Cell;
use std::collections::HashSet;
use std::sync::Arc;

use zyron_catalog::{
    ColumnId, ConstraintEntry, ConstraintType, ReferentialAction, TableEntry, TableId,
};
use zyron_common::page::PageId;
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::LiteralValue;
use zyron_planner::binder::{BoundAssignment, BoundExpr};
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{HeapPage, TupleId};

use crate::batch::{ColumnBuilder, DataBatch, create_builders, decode_tuple_into_builders};
use crate::context::ExecutionContext;
use crate::operator::modify::{DeleteOperator, UpdateOperator, encode_btree_key_into};
use crate::operator::scan::read_page_through_pool;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

thread_local! {
    /// Cascade recursion depth guard. A foreign-key cycle with cascading
    /// actions would otherwise recurse without bound, so a delete or update
    /// that drives child mutations past this depth is rejected.
    static CASCADE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

const MAX_CASCADE_DEPTH: u32 = 64;

/// Resolves a constraint's column-id list to positions in a table's column
/// vector. Returns None if any column is absent (a corrupt constraint), so the
/// caller can skip rather than enforce against a wrong layout.
fn column_positions(table: &TableEntry, columns: &[ColumnId]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(columns.len());
    for cid in columns {
        let pos = table.columns.iter().position(|c| c.id == *cid)?;
        out.push(pos);
    }
    Some(out)
}

/// Encodes a composite key from the given column positions of one batch row,
/// length-prefixing each component so multi-column keys are unambiguous.
/// Returns None when any component is null (MATCH SIMPLE skips the row).
fn encode_composite_key(
    batch: &DataBatch,
    row: usize,
    positions: &[usize],
    types: &[TypeId],
) -> Option<Vec<u8>> {
    let mut key = Vec::with_capacity(positions.len() * 12);
    let mut scratch = Vec::with_capacity(16);
    for (i, &pos) in positions.iter().enumerate() {
        if !encode_btree_key_into(batch, row, pos, types[i], &mut scratch) {
            return None;
        }
        key.extend_from_slice(&(scratch.len() as u16).to_le_bytes());
        key.extend_from_slice(&scratch);
    }
    Some(key)
}

/// Decodes one heap tuple's bytes into a single-row batch over a table's full
/// column set so key columns can be read back by position.
fn decode_tuple_to_batch(data: &[u8], table: &TableEntry) -> DataBatch {
    let column_to_builder: Vec<Option<u16>> =
        (0..table.columns.len()).map(|i| Some(i as u16)).collect();
    let mut builders: Vec<ColumnBuilder> = table
        .columns
        .iter()
        .map(|c| {
            let phys = TypeId::timestamp_physical_type_id(c.type_id, c.ts_precision);
            if phys != c.type_id {
                ColumnBuilder::new_ts(c.type_id, phys, c.ts_precision, 1)
            } else {
                ColumnBuilder::new(c.type_id, 1)
            }
        })
        .collect();
    decode_tuple_into_builders(data, &table.columns, &column_to_builder, &mut builders);
    DataBatch::new(builders.into_iter().map(|b| b.finish()).collect())
}

/// Returns the B+tree index id whose leading column is `col_id` for the table,
/// or None. Used to take the O(1) probe path for single-column keys.
fn leading_btree_for_column(
    ctx: &ExecutionContext,
    table_id: u32,
    col_id: ColumnId,
) -> Option<u32> {
    let snap = ctx.index_snapshot_for_table(table_id);
    snap.btree
        .iter()
        .find(|(_, c, _)| *c == col_id)
        .map(|(id, _, _)| id.0)
}

/// Reads the heap tuple at `tid` and returns its decoded single-row batch when
/// the tuple is live and visible under the statement snapshot, else None.
async fn read_visible_tuple(
    ctx: &ExecutionContext,
    table: &TableEntry,
    tid: TupleId,
) -> Result<Option<DataBatch>> {
    let page_data =
        read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, tid.page_id).await?;
    let page = HeapPage::from_bytes(page_data);
    let Some(view) = page.get_tuple_view(zyron_storage::SlotId(tid.slot_id)) else {
        return Ok(None);
    };
    if view.is_deleted() || !view.header.is_visible_to(&ctx.snapshot) {
        return Ok(None);
    }
    Ok(Some(decode_tuple_to_batch(view.data, table)))
}

/// Scans a table's heap once and collects the composite keys of every live,
/// visible row over `positions`. Used as the existence-check fallback when no
/// usable index exists and as the match source for parent-side enforcement.
async fn collect_visible_keys(
    ctx: &ExecutionContext,
    table: &TableEntry,
    positions: &[usize],
    types: &[TypeId],
) -> Result<HashSet<Vec<u8>>> {
    let mut keys = HashSet::new();
    let heap = ctx.get_heap_file(table.id).await?;
    let num_pages = heap.num_pages_cached() as u32;
    for page_num in 0..num_pages {
        ctx.check_cancelled()?;
        let page_id = PageId::new(table.heap_file_id, page_num as u64);
        let page_data =
            read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, page_id).await?;
        let header = HeapPage::heap_header_from_slice(&page_data);
        if header.slot_count == 0 {
            continue;
        }
        let page = HeapPage::from_bytes(page_data);
        for slot in 0..header.slot_count {
            let Some(view) = page.get_tuple_view(zyron_storage::SlotId(slot)) else {
                continue;
            };
            if view.is_deleted() || !view.header.is_visible_to(&ctx.snapshot) {
                continue;
            }
            let batch = decode_tuple_to_batch(view.data, table);
            if let Some(key) = encode_composite_key(&batch, 0, positions, types) {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

/// Probes whether a single visible parent row carries the referenced key. Uses
/// the parent index for single-column keys, otherwise the cached key set.
async fn parent_key_exists(
    ctx: &ExecutionContext,
    parent: &TableEntry,
    parent_positions: &[usize],
    parent_types: &[TypeId],
    child_key: &[u8],
    single_col_index: Option<u32>,
    fallback_keys: &Option<HashSet<Vec<u8>>>,
    single_col_raw: Option<&[u8]>,
) -> Result<bool> {
    if let (Some(idx_id), Some(raw)) = (single_col_index, single_col_raw) {
        if let Some(btree) = ctx.get_index(zyron_catalog::IndexId(idx_id)) {
            // Index keys are composite (value followed by a tuple-id suffix), so
            // probe the whole key range for this value and collect candidate
            // tids; multiple rows may share the value.
            let suffix = crate::operator::modify::INDEX_TID_SUFFIX_LEN;
            let mut lo = raw.to_vec();
            lo.extend(std::iter::repeat(0u8).take(suffix));
            let mut hi = raw.to_vec();
            hi.extend(std::iter::repeat(0xFFu8).take(suffix));
            let mut candidates: Vec<TupleId> = Vec::new();
            btree.range_scan_for_each(Some(&lo), Some(&hi), |_k, tid| {
                candidates.push(tid);
                true
            });
            for tid in candidates {
                // Stored tids are normalized to file id 0; resolve to the parent
                // heap file before reading.
                let resolved = TupleId::new(
                    PageId::new(parent.heap_file_id, tid.page_id.page_num),
                    tid.slot_id,
                );
                if let Some(batch) = read_visible_tuple(ctx, parent, resolved).await? {
                    if let Some(found) =
                        encode_composite_key(&batch, 0, parent_positions, parent_types)
                    {
                        if found == child_key {
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }
    if let Some(keys) = fallback_keys {
        return Ok(keys.contains(child_key));
    }
    Ok(false)
}

/// Enforces child-side foreign keys for a batch of inserted or updated rows.
/// The batch columns are in table-column order (the layout produced by the
/// scan and insert paths).
pub async fn check_child_fks(
    ctx: &ExecutionContext,
    table: &TableEntry,
    batch: &DataBatch,
) -> Result<()> {
    let num_rows = batch.columns.first().map(|c| c.len()).unwrap_or(0);
    if num_rows == 0 {
        return Ok(());
    }

    for con in &table.constraints {
        if con.constraint_type != ConstraintType::ForeignKey {
            continue;
        }
        let Some(parent_id) = con.ref_table_id else {
            continue;
        };
        let Some(local_pos) = column_positions(table, &con.columns) else {
            continue;
        };
        let parent = ctx.catalog.get_table_by_id(parent_id)?;
        let Some(parent_pos) = column_positions(&parent, &con.ref_columns) else {
            continue;
        };
        let local_types: Vec<TypeId> = local_pos
            .iter()
            .map(|&p| table.columns[p].type_id)
            .collect();
        let parent_types: Vec<TypeId> = parent_pos
            .iter()
            .map(|&p| parent.columns[p].type_id)
            .collect();

        // Single-column FK whose parent column is indexed takes the probe path.
        let single_idx = if con.ref_columns.len() == 1 {
            leading_btree_for_column(ctx, parent.id.0, con.ref_columns[0])
        } else {
            None
        };

        // Build the parent key set once when no index probe is available.
        let fallback_keys = if single_idx.is_none() {
            Some(collect_visible_keys(ctx, &parent, &parent_pos, &parent_types).await?)
        } else {
            None
        };

        let mut raw_scratch = Vec::with_capacity(16);
        for row in 0..num_rows {
            // MATCH SIMPLE: any null FK component skips the row.
            if local_pos.iter().any(|&p| batch.columns[p].is_null(row)) {
                continue;
            }
            let Some(child_key) = encode_composite_key(batch, row, &local_pos, &local_types) else {
                continue;
            };
            // Raw single-column key (no length prefix) matches the index layout.
            let single_raw = if single_idx.is_some() {
                raw_scratch.clear();
                encode_btree_key_into(batch, row, local_pos[0], local_types[0], &mut raw_scratch);
                Some(raw_scratch.as_slice())
            } else {
                None
            };
            let exists = parent_key_exists(
                ctx,
                &parent,
                &parent_pos,
                &parent_types,
                &child_key,
                single_idx,
                &fallback_keys,
                single_raw,
            )
            .await?;
            if !exists {
                return Err(ZyronError::ForeignKeyViolation(format!(
                    "insert or update on table \"{}\" violates foreign key constraint \"{}\": referenced key not present in \"{}\"",
                    table.name, con.name, parent.name
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parent-side enforcement (ON DELETE / ON UPDATE)
// ---------------------------------------------------------------------------

/// Single-shot source operator that yields one pre-built batch of rows with
/// their tuple ids. Used to feed gathered child rows into a Delete or Update
/// operator so cascades reuse the standard mutation and index-maintenance path.
struct VecSourceOperator {
    batch: Option<ExecutionBatch>,
}

impl Operator for VecSourceOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.batch.take()) })
    }
}

/// Builds a logical schema mirroring a table's full column list, used as the
/// input schema when driving an Update operator over gathered child rows.
fn logical_schema(table: &TableEntry) -> Vec<LogicalColumn> {
    table
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: None,
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            ts_precision: c.ts_precision,
        })
        .collect()
}

/// Scans a child table once and gathers every live, visible row whose foreign
/// key matches one of `target_keys`, returning the rows as a batch plus their
/// tuple ids. A full scan is used because child foreign keys are typically
/// non-unique, so an index probe would miss duplicate references.
async fn gather_matching_children(
    ctx: &ExecutionContext,
    child: &TableEntry,
    fk_positions: &[usize],
    fk_types: &[TypeId],
    target_keys: &HashSet<Vec<u8>>,
) -> Result<(DataBatch, Vec<TupleId>)> {
    let schema = logical_schema(child);
    let mut builders = create_builders(&schema, 0);
    let mut tuple_ids: Vec<TupleId> = Vec::new();

    let heap = ctx.get_heap_file(child.id).await?;
    let num_pages = heap.num_pages_cached() as u32;
    for page_num in 0..num_pages {
        ctx.check_cancelled()?;
        let page_id = PageId::new(child.heap_file_id, page_num as u64);
        let page_data =
            read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, page_id).await?;
        let header = HeapPage::heap_header_from_slice(&page_data);
        if header.slot_count == 0 {
            continue;
        }
        let page = HeapPage::from_bytes(page_data);
        for slot in 0..header.slot_count {
            let Some(view) = page.get_tuple_view(zyron_storage::SlotId(slot)) else {
                continue;
            };
            if view.is_deleted() || !view.header.is_visible_to(&ctx.snapshot) {
                continue;
            }
            let row = decode_tuple_to_batch(view.data, child);
            // MATCH SIMPLE: a row with any null FK component references nothing.
            if fk_positions.iter().any(|&p| row.columns[p].is_null(0)) {
                continue;
            }
            let Some(key) = encode_composite_key(&row, 0, fk_positions, fk_types) else {
                continue;
            };
            if target_keys.contains(&key) {
                for (b, col) in builders.iter_mut().zip(&row.columns) {
                    b.push(&col.get_scalar(0));
                }
                tuple_ids.push(TupleId::new(
                    PageId::new(child.heap_file_id, page_num as u64),
                    slot,
                ));
            }
        }
    }
    let batch = DataBatch::new(builders.into_iter().map(|b| b.finish()).collect());
    Ok((batch, tuple_ids))
}

/// Drives a sub-operator to completion, discarding its output batches.
async fn drive_to_completion(mut op: Box<dyn Operator>) -> Result<()> {
    while op.next().await?.is_some() {}
    Ok(())
}

/// Enforces ON DELETE actions for every foreign key that references `parent`
/// when the rows in `old_batch` are about to be deleted. Restrict and NoAction
/// reject the delete when matching child rows exist; Cascade deletes them;
/// SetNull rewrites the child foreign-key columns to null.
pub async fn enforce_parent_delete(
    ctx: &Arc<ExecutionContext>,
    parent: &TableEntry,
    old_batch: &DataBatch,
) -> Result<()> {
    let referencing = ctx.catalog.referencing_constraints(parent.id);
    if referencing.is_empty() {
        return Ok(());
    }
    let num_rows = old_batch.columns.first().map(|c| c.len()).unwrap_or(0);
    if num_rows == 0 {
        return Ok(());
    }

    for (child, con) in referencing {
        let Some(parent_pos) = column_positions(parent, &con.ref_columns) else {
            continue;
        };
        let Some(child_pos) = column_positions(&child, &con.columns) else {
            continue;
        };
        let parent_types: Vec<TypeId> = parent_pos
            .iter()
            .map(|&p| parent.columns[p].type_id)
            .collect();
        let child_types: Vec<TypeId> = child_pos
            .iter()
            .map(|&p| child.columns[p].type_id)
            .collect();

        // Keys of the parent rows being deleted (null components reference no
        // child, so they are skipped).
        let mut target_keys = HashSet::new();
        for row in 0..num_rows {
            if parent_pos
                .iter()
                .any(|&p| old_batch.columns[p].is_null(row))
            {
                continue;
            }
            if let Some(k) = encode_composite_key(old_batch, row, &parent_pos, &parent_types) {
                target_keys.insert(k);
            }
        }
        if target_keys.is_empty() {
            continue;
        }

        let (matched, tuple_ids) =
            gather_matching_children(ctx, &child, &child_pos, &child_types, &target_keys).await?;
        if tuple_ids.is_empty() {
            continue;
        }

        match con.on_delete {
            ReferentialAction::NoAction | ReferentialAction::Restrict => {
                return Err(ZyronError::ForeignKeyViolation(format!(
                    "delete on table \"{}\" violates foreign key constraint \"{}\" on \"{}\": {} dependent row(s) remain",
                    parent.name,
                    con.name,
                    child.name,
                    tuple_ids.len()
                )));
            }
            ReferentialAction::Cascade => {
                run_with_depth_guard(async {
                    let source = Box::new(VecSourceOperator {
                        batch: Some(ExecutionBatch::with_tuple_ids(matched, tuple_ids)),
                    });
                    let del = Box::new(DeleteOperator::new(source, Arc::clone(ctx), child.id));
                    drive_to_completion(del).await
                })
                .await?;
            }
            ReferentialAction::SetNull => {
                run_with_depth_guard(async {
                    drive_set_null(ctx, &child, &con, matched, tuple_ids).await
                })
                .await?;
            }
            ReferentialAction::SetDefault => {
                return Err(ZyronError::ForeignKeyViolation(format!(
                    "ON DELETE SET DEFAULT on constraint \"{}\" requires an evaluable column default, which is not available",
                    con.name
                )));
            }
        }
    }
    Ok(())
}

/// Runs a cascading mutation under the recursion-depth guard so a foreign-key
/// cycle cannot recurse without bound.
async fn run_with_depth_guard<F>(fut: F) -> Result<()>
where
    F: std::future::Future<Output = Result<()>>,
{
    let depth = CASCADE_DEPTH.with(|d| {
        let v = d.get() + 1;
        d.set(v);
        v
    });
    let result = if depth > MAX_CASCADE_DEPTH {
        Err(ZyronError::ForeignKeyViolation(format!(
            "foreign key cascade exceeded maximum depth {MAX_CASCADE_DEPTH}, possible cycle"
        )))
    } else {
        fut.await
    };
    CASCADE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    result
}

/// Updates the gathered child rows, setting each foreign-key column to null.
async fn drive_set_null(
    ctx: &Arc<ExecutionContext>,
    child: &TableEntry,
    con: &ConstraintEntry,
    matched: DataBatch,
    tuple_ids: Vec<TupleId>,
) -> Result<()> {
    let assignments: Vec<BoundAssignment> = con
        .columns
        .iter()
        .filter_map(|cid| {
            child
                .columns
                .iter()
                .find(|c| c.id == *cid)
                .map(|c| BoundAssignment {
                    column_id: c.id,
                    value: BoundExpr::Literal {
                        value: LiteralValue::Null,
                        type_id: c.type_id,
                    },
                })
        })
        .collect();
    let source = Box::new(VecSourceOperator {
        batch: Some(ExecutionBatch::with_tuple_ids(matched, tuple_ids)),
    });
    // Enforce the child's CHECK constraints on the cascaded SET NULL image so a
    // column that must stay non-null (or otherwise constrained) blocks the
    // cascade rather than silently writing a violating row.
    let checks = zyron_planner::bind_table_check_constraints(&ctx.catalog, child).await?;
    let upd = Box::new(UpdateOperator::new(
        source,
        Arc::clone(ctx),
        child.id,
        assignments,
        logical_schema(child),
        checks,
    ));
    drive_to_completion(upd).await
}

/// Enforces ON UPDATE actions when a parent row's referenced key changes.
/// Restrict and NoAction reject the change when dependents exist; Cascade
/// rewrites the child key to the new value; SetNull nulls the child key.
pub async fn enforce_parent_update(
    ctx: &Arc<ExecutionContext>,
    parent: &TableEntry,
    old_batch: &DataBatch,
    new_batch: &DataBatch,
) -> Result<()> {
    let referencing = ctx.catalog.referencing_constraints(parent.id);
    if referencing.is_empty() {
        return Ok(());
    }
    let num_rows = old_batch.columns.first().map(|c| c.len()).unwrap_or(0);
    if num_rows == 0 {
        return Ok(());
    }

    for (child, con) in referencing {
        let Some(parent_pos) = column_positions(parent, &con.ref_columns) else {
            continue;
        };
        let Some(child_pos) = column_positions(&child, &con.columns) else {
            continue;
        };
        let parent_types: Vec<TypeId> = parent_pos
            .iter()
            .map(|&p| parent.columns[p].type_id)
            .collect();
        let child_types: Vec<TypeId> = child_pos
            .iter()
            .map(|&p| child.columns[p].type_id)
            .collect();

        // Only rows whose referenced key actually changed have any effect.
        let mut changed_keys = HashSet::new();
        for row in 0..num_rows {
            let old_key = encode_composite_key(old_batch, row, &parent_pos, &parent_types);
            let new_key = encode_composite_key(new_batch, row, &parent_pos, &parent_types);
            if let Some(ok) = old_key {
                if Some(&ok) != new_key.as_ref() {
                    changed_keys.insert(ok);
                }
            }
        }
        if changed_keys.is_empty() {
            continue;
        }

        let (matched, tuple_ids) =
            gather_matching_children(ctx, &child, &child_pos, &child_types, &changed_keys).await?;
        if tuple_ids.is_empty() {
            continue;
        }

        match con.on_update {
            ReferentialAction::NoAction | ReferentialAction::Restrict => {
                return Err(ZyronError::ForeignKeyViolation(format!(
                    "update on table \"{}\" violates foreign key constraint \"{}\" on \"{}\": {} dependent row(s) reference the old key",
                    parent.name,
                    con.name,
                    child.name,
                    tuple_ids.len()
                )));
            }
            ReferentialAction::SetNull => {
                run_with_depth_guard(async {
                    drive_set_null(ctx, &child, &con, matched, tuple_ids).await
                })
                .await?;
            }
            ReferentialAction::Cascade | ReferentialAction::SetDefault => {
                return Err(ZyronError::ForeignKeyViolation(format!(
                    "ON UPDATE {} on constraint \"{}\" is not supported for multi-row key remapping",
                    if con.on_update == ReferentialAction::Cascade {
                        "CASCADE"
                    } else {
                        "SET DEFAULT"
                    },
                    con.name
                )));
            }
        }
    }
    Ok(())
}

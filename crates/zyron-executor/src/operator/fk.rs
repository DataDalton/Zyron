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
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use zyron_catalog::{ColumnId, ConstraintEntry, ConstraintType, ReferentialAction, TableEntry};
use zyron_common::page::PageId;
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::LiteralValue;
use zyron_planner::binder::{BoundAssignment, BoundExpr};
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{HeapPage, TupleId};

use crate::batch::{ColumnBuilder, DataBatch, create_builders, decode_tuple_into_builders};
use crate::column::ScalarValue;
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
pub(crate) fn decode_tuple_to_batch(data: &[u8], table: &TableEntry) -> DataBatch {
    let column_to_builder: Vec<Option<u16>> =
        (0..table.columns.len()).map(|i| Some(i as u16)).collect();
    let mut builders: Vec<ColumnBuilder> = table
        .columns
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
    decode_tuple_into_builders(data, &table.columns, &column_to_builder, &mut builders);
    DataBatch::new(builders.into_iter().map(|b| b.finish()).collect())
}

/// Returns the B+tree index id whose leading column is `col_id` for the table,
/// or None. Used to take the O(1) probe path for single-column keys.
///
/// An index over more stores the further components between the value and
/// the locator suffix, so the probe reads them as part of the key. A
/// leading-column range still brackets every entry for the value, and the
/// probe re-reads each candidate row to confirm the key, so the extra
/// components cost candidates rather than correctness.
fn leading_btree_for_column(
    ctx: &ExecutionContext,
    table_id: u32,
    col_id: ColumnId,
) -> Option<u32> {
    let snap = ctx.index_snapshot_for_table(table_id);
    snap.btree
        .iter()
        .find(|spec| spec.leading() == col_id)
        .map(|spec| spec.id.0)
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

/// Collects the composite keys of every live, visible row over `positions`,
/// from the heap and from any columnar segments. Used as the existence-check
/// fallback when no usable index exists and as the match source for
/// parent-side enforcement.
/// Adds the keys of rows a branch appended to a table, from the branch's own
/// append file.
///
/// Separate from the main walk so its locals do not enlarge the foreign key
/// check's future, which the cascade paths already nest under an update
/// operator.
#[allow(clippy::too_many_arguments)]
/// Walks the rows a branch appended to its own file, which no main page
/// holds and so no main page walk can reach
async fn for_each_append_row(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    append_file_id: u32,
    append_pages: u64,
    f: &mut (dyn FnMut(&DataBatch, usize) + Send),
) -> Result<()> {
    for page_num in 0..append_pages {
        ctx.check_cancelled()?;
        let page_data = read_page_through_pool(
            &ctx.buffer_pool,
            &ctx.disk_manager,
            PageId::new(append_file_id, page_num),
        )
        .await?;
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
            f(&batch, 0);
        }
    }
    Ok(())
}

async fn collect_visible_keys(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    positions: &[usize],
    types: &[TypeId],
) -> Result<HashSet<Vec<u8>>> {
    let mut keys = HashSet::new();
    for_each_visible_row(ctx, table, &mut |batch, row| {
        if let Some(key) = encode_composite_key(batch, row, positions, types) {
            keys.insert(key);
        }
    })
    .await?;
    Ok(keys)
}

/// Walks every row of a table this statement can see, once.
///
/// Three places hold rows and each has its own visibility rule: the main
/// heap pages under the snapshot, the append file a branch writes its own
/// rows into, and the columnar segments a fold moved rows into. A collector
/// built on top of this sees all three, which is what keeps a parent probe
/// from answering for the main line while a branch holds different rows.
async fn for_each_visible_row(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    f: &mut (dyn FnMut(&DataBatch, usize) + Send),
) -> Result<()> {
    let heap = ctx.get_heap_file(table.id).await?;
    let num_pages = heap.num_pages_cached() as u32;
    // Under a branch the parent's keys are its main rows as the branch sees
    // them, plus the rows the branch appended. Reading main alone would
    // refuse a reference to a parent the branch itself added, and admit one
    // to a parent the branch deleted
    let branch = ctx.active_branch_id;
    let (append_file_id, append_pages) = match (branch, ctx.branch_catalog.as_ref()) {
        (Some(bid), Some(cat)) => {
            let files = cat.branch_files_for(bid, table.heap_file_id);
            (
                Some(files.append_file_id),
                cat.append_page_count(bid, table.heap_file_id),
            )
        }
        _ => (None, 0),
    };
    for page_num in 0..num_pages {
        ctx.check_cancelled()?;
        let page_id =
            ctx.resolve_branch_page(branch, PageId::new(table.heap_file_id, page_num as u64));
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
            f(&batch, 0);
        }
    }

    // The branch's own appended rows, which live in its append file and so
    // are in no main page above. Boxed because this runs inside the foreign
    // key check, which the cascade paths nest under an update operator, and
    // an inlined frame here is what exhausts the stack in a debug build
    if let Some(file_id) = append_file_id {
        Box::pin(for_each_append_row(ctx, table, file_id, append_pages, f)).await?;
    }

    // Folded rows live in columnar segments the heap walk cannot see. Drain
    // a columnar scan (patch overlay and MVCC visibility applied by the
    // operator) so a folded parent row still proves existence
    if !table.columnar.segments.is_empty() {
        let logical: Vec<LogicalColumn> = table
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
            .collect();
        let mut op = crate::operator::column_scan::ColumnScanOperator::new_for_dml(
            Arc::clone(ctx),
            table.id,
            logical,
            None,
        )?;
        while let Some(eb) = op.next().await? {
            for row in 0..eb.batch.num_rows {
                f(&eb.batch, row);
            }
        }
    }
    Ok(())
}

/// Every parent period that shares a scalar key, keyed by that key.
///
/// A period foreign key matches its last column pair by containment rather
/// than equality, so the probe needs the parent's periods themselves and not
/// just the fact that a key exists.
async fn collect_visible_key_periods(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    scalar_positions: &[usize],
    scalar_types: &[TypeId],
    period_position: usize,
) -> Result<HashMap<Vec<u8>, Vec<Vec<u8>>>> {
    let mut out: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();
    for_each_visible_row(ctx, table, &mut |batch, row| {
        let Some(key) = encode_composite_key(batch, row, scalar_positions, scalar_types) else {
            return;
        };
        let Some(column) = batch.columns.get(period_position) else {
            return;
        };
        if column.is_null(row) {
            return;
        }
        if let ScalarValue::Binary(period) = column.get_scalar(row) {
            out.entry(key).or_default().push(period);
        }
    })
    .await?;
    Ok(out)
}

/// Whether the parent's periods together cover `child`.
///
/// SQL states temporal referential integrity over the union of the matching
/// parent rows, not over any single one, so two adjacent parent periods
/// covering a child period between them satisfy the constraint. The parent
/// periods are merged in position order and the child is tested against each
/// merged run, which is what makes the check read the union rather than a
/// row at a time.
fn periods_cover(parents: &[Vec<u8>], child: &[u8]) -> bool {
    use zyron_types::range::{
        RANGE_ELEM_SIZE, range_adjacent, range_contains_range, range_index_key, range_is_empty,
        range_overlaps, range_union,
    };

    if range_is_empty(child) {
        // An empty period demands nothing of the parent
        return true;
    }
    let mut ordered: Vec<&Vec<u8>> = parents.iter().filter(|p| !range_is_empty(p)).collect();
    if ordered.is_empty() {
        return false;
    }
    // The index form sorts by position, which is the order a coverage sweep
    // needs, and it is the same ordering the temporal constraint maintains
    ordered.sort_by_cached_key(|p| range_index_key(p, RANGE_ELEM_SIZE));

    let mut run: Vec<u8> = ordered[0].to_vec();
    for next in &ordered[1..] {
        if range_contains_range(&run, child, RANGE_ELEM_SIZE) {
            return true;
        }
        let joins = range_overlaps(&run, next, RANGE_ELEM_SIZE)
            || range_adjacent(&run, next, RANGE_ELEM_SIZE);
        run = if joins {
            match range_union(&run, next, RANGE_ELEM_SIZE) {
                Ok(merged) => merged,
                // Two periods that neither meet nor overlap start a new run
                Err(_) => next.to_vec(),
            }
        } else {
            next.to_vec()
        };
    }
    range_contains_range(&run, child, RANGE_ELEM_SIZE)
}

/// The child side of a period foreign key: every row's period has to be
/// covered by the periods of the parent rows sharing its scalar key.
///
/// The parent's periods are read once per statement through the same visible
/// row walk the plain key probe uses, so a branch's appended rows and its
/// deletions both count, and a folded columnar parent row counts too.
#[allow(clippy::too_many_arguments)]
async fn check_child_period_fk(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    parent: &TableEntry,
    con: &ConstraintEntry,
    batch: &DataBatch,
    local_pos: &[usize],
    parent_pos: &[usize],
    violations: &mut FkViolations,
) -> Result<()> {
    // The period is the last column of each side, and the scalar key is
    // everything before it
    let Some((&child_period_pos, child_scalar_pos)) = local_pos.split_last() else {
        return Ok(());
    };
    let Some((&parent_period_pos, parent_scalar_pos)) = parent_pos.split_last() else {
        return Ok(());
    };
    if table.columns[child_period_pos].type_id != TypeId::Range
        || parent.columns[parent_period_pos].type_id != TypeId::Range
    {
        return Err(zyron_common::ZyronError::PlanError(format!(
            "foreign key \"{}\" declares PERIOD but its last column pair is not a range",
            con.name
        )));
    }
    let child_scalar_types: Vec<TypeId> = child_scalar_pos
        .iter()
        .map(|&p| table.columns[p].type_id)
        .collect();
    let parent_scalar_types: Vec<TypeId> = parent_scalar_pos
        .iter()
        .map(|&p| parent.columns[p].type_id)
        .collect();

    let parent_periods = collect_visible_key_periods(
        ctx,
        parent,
        parent_scalar_pos,
        &parent_scalar_types,
        parent_period_pos,
    )
    .await?;

    let num_rows = batch.columns.first().map(|c| c.len()).unwrap_or(0);
    for row in 0..num_rows {
        // MATCH SIMPLE: a null anywhere in the key leaves the row
        // unconstrained, and a null period references nothing
        if child_scalar_pos
            .iter()
            .any(|&p| batch.columns[p].is_null(row))
            || batch.columns[child_period_pos].is_null(row)
        {
            continue;
        }
        let Some(key) = encode_composite_key(batch, row, child_scalar_pos, &child_scalar_types)
        else {
            continue;
        };
        let ScalarValue::Binary(child_period) = batch.columns[child_period_pos].get_scalar(row)
        else {
            continue;
        };
        let covered = parent_periods
            .get(&key)
            .is_some_and(|periods| periods_cover(periods, &child_period));
        if !covered {
            violations.record(con, row, &table.name, &parent.name)?;
        }
    }
    Ok(())
}

/// Probes whether a single visible parent row carries the referenced key. Uses
/// the parent index for single-column keys, otherwise the cached key set.
async fn parent_key_exists(
    ctx: &Arc<ExecutionContext>,
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
            // A stored key is the indexed value followed by a locator suffix,
            // so probe the whole key range for this value and collect the
            // candidates; multiple rows may share the value.
            let lo = raw.to_vec();
            let hi = crate::operator::modify::index_key_upper_bound(raw);
            let mut heap_candidates: Vec<TupleId> = Vec::new();
            let mut columnar_candidates: Vec<zyron_common::RowLocator> = Vec::new();
            btree.range_scan_for_each(Some(&lo), hi.as_deref(), |_k, loc| {
                match loc {
                    // stored heap locators are normalized to file id 0,
                    // resolve to the parent heap file before reading
                    zyron_common::RowLocator::Heap { page, slot } => {
                        heap_candidates.push(TupleId::new(
                            PageId::new(parent.heap_file_id, page.page_num),
                            slot,
                        ));
                    }
                    loc @ zyron_common::RowLocator::Columnar { .. } => {
                        columnar_candidates.push(loc);
                    }
                    zyron_common::RowLocator::Lake { .. } => {}
                }
                true
            });
            for resolved in heap_candidates {
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
            if !columnar_candidates.is_empty() {
                // One batched fetch of the candidates' current visible values,
                // then re-encode and compare. An updated columnar row keeps
                // its locator, so a stale-value entry must not prove existence
                let logical: Vec<LogicalColumn> = parent
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
                    .collect();
                let fetcher = crate::operator::doc_fetch::DocRowFetcher::prepare_columnar_only(
                    ctx,
                    parent.id,
                    &logical,
                    &columnar_candidates,
                    None,
                )
                .await?;
                for loc in &columnar_candidates {
                    let Some((file_id, sys_rowid)) = loc.columnar_pair() else {
                        continue;
                    };
                    let Some(vals) = fetcher.columnar_row(file_id, sys_rowid) else {
                        continue;
                    };
                    let mut builders: Vec<ColumnBuilder> = parent
                        .columns
                        .iter()
                        .map(|c| {
                            let phys =
                                TypeId::timestamp_physical_type_id(c.type_id, c.fractional_digits);
                            if phys != c.type_id || c.fractional_digits.is_some() {
                                ColumnBuilder::new_ts(c.type_id, phys, c.fractional_digits, 1)
                            } else {
                                ColumnBuilder::new(c.type_id, 1)
                            }
                        })
                        .collect();
                    for (b, v) in builders.iter_mut().zip(vals.iter()) {
                        b.push(v);
                    }
                    let row = DataBatch::new(builders.into_iter().map(|b| b.finish()).collect());
                    if let Some(found) =
                        encode_composite_key(&row, 0, parent_positions, parent_types)
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
/// Rows a foreign key rejected, and where they go.
///
/// A constraint declared ON VIOLATION QUARANTINE diverts its rejected rows
/// instead of aborting, which is what keeps a bulk load usable: the rows
/// that reference real parents land, the rest are preserved for inspection.
/// Every other constraint fails the statement on its first rejection, so
/// this collects nothing for them.
#[derive(Debug, Default)]
pub struct FkViolations {
    /// Row index paired with the quarantine table it belongs in and the
    /// constraint that rejected it
    rows: Vec<(usize, u32, String)>,
}

impl FkViolations {
    /// Records one rejected row, or fails the statement when the constraint
    /// does not quarantine.
    fn record(
        &mut self,
        con: &zyron_catalog::schema::ConstraintEntry,
        row: usize,
        table_name: &str,
        parent_name: &str,
    ) -> Result<()> {
        use zyron_catalog::schema::ConstraintViolationAction;
        match (con.on_violation, con.quarantine_table_id) {
            (ConstraintViolationAction::Quarantine, Some(quarantine_id)) => {
                self.rows.push((row, quarantine_id, con.name.clone()));
                Ok(())
            }
            // Declared to quarantine with nowhere to put the row: refusing is
            // the only honest answer, dropping it would lose data silently
            (ConstraintViolationAction::Quarantine, None) => {
                Err(ZyronError::ForeignKeyViolation(format!(
                    "foreign key constraint \"{}\" on \"{}\" quarantines violations but has no \
                     quarantine table",
                    con.name, table_name
                )))
            }
            _ => Err(ZyronError::ForeignKeyViolation(format!(
                "insert or update on table \"{}\" violates foreign key constraint \"{}\": \
                 row {} references a key not present in \"{}\"",
                table_name, con.name, row, parent_name
            ))),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Rejected rows grouped by quarantine table, each with the constraint
    /// names that rejected it. A row rejected by two constraints is written
    /// once, naming both.
    pub fn by_table(&self) -> Vec<(u32, Vec<usize>, Vec<String>)> {
        let mut grouped: std::collections::BTreeMap<
            u32,
            std::collections::BTreeMap<usize, Vec<String>>,
        > = std::collections::BTreeMap::new();
        for (row, table_id, name) in &self.rows {
            grouped
                .entry(*table_id)
                .or_default()
                .entry(*row)
                .or_default()
                .push(name.clone());
        }
        grouped
            .into_iter()
            .map(|(table_id, rows)| {
                let mut indexes = Vec::with_capacity(rows.len());
                let mut names = Vec::with_capacity(rows.len());
                for (row, mut reasons) in rows {
                    reasons.sort();
                    reasons.dedup();
                    indexes.push(row);
                    names.push(reasons.join(", "));
                }
                (table_id, indexes, names)
            })
            .collect()
    }

    /// Fails when anything was rejected, used where diverting a row is not
    /// possible.
    ///
    /// An update cannot quarantine: the row is already in the table, so
    /// moving it aside would delete it. A quarantining constraint therefore
    /// still refuses an update that would break it.
    pub fn deny_diversion(&self, table_name: &str) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let names: Vec<&str> = self.rows.iter().map(|(_, _, name)| name.as_str()).collect();
        Err(ZyronError::ForeignKeyViolation(format!(
            "update on table \"{}\" violates foreign key constraint \"{}\": an update cannot              be quarantined, the row is already in the table",
            table_name,
            names.first().copied().unwrap_or("")
        )))
    }

    /// Every rejected row index, ascending and deduplicated.
    pub fn rows(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.rows.iter().map(|(row, _, _)| *row).collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Checks a foreign key whose parent is a lake table.
///
/// The child's referencing cells are handed over under the parent's column
/// ids, so the lake probe reads the parent's own key columns and prunes
/// files by their recorded bounds and value blooms before opening any.
fn check_lake_parent(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    parent: &TableEntry,
    con: &zyron_catalog::schema::ConstraintEntry,
    batch: &DataBatch,
    local_pos: &[usize],
    violations: &mut FkViolations,
) -> Result<()> {
    let paths = zyron_lake::LakePaths::new(ctx.disk_manager.data_dir(), parent.id.0);
    // The probe reads the session's effective head, so a parent row the
    // branch inserted satisfies the reference and one the branch deleted
    // no longer does
    let head = crate::operator::lake_scan::effective_head(ctx, None);
    let log = crate::operator::lake_scan::open_lake_head(&paths, &parent.name, head)?;
    let manifest = log.latest_manifest()?;

    let num_rows = batch.columns.first().map(|c| c.len()).unwrap_or(0);
    let parent_ids: Vec<u32> = con.ref_columns.iter().map(|c| c.0 as u32).collect();
    let mut values: Vec<zyron_lake::ColumnData> = Vec::with_capacity(parent_ids.len());
    for (i, parent_id) in parent_ids.iter().enumerate() {
        let child_col = &table.columns[local_pos[i]];
        let value_size = child_col.physical_type_id().fixed_size().unwrap_or(0);
        let column = &batch.columns[local_pos[i]];
        let mut cells = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let scalar = column.get_scalar(row);
            cells.push(match scalar {
                crate::column::ScalarValue::Null => None,
                ref v => Some(crate::batch::encode_scalar_value(
                    child_col.type_id,
                    v,
                    value_size,
                )),
            });
        }
        values.push(zyron_lake::ColumnData::from_cells(*parent_id, cells));
    }

    let (outcome, _) = zyron_lake::check_foreign_key(log.paths(), &manifest, &parent_ids, &values)?;
    match outcome {
        zyron_lake::ForeignKeyOutcome::Ok => Ok(()),
        zyron_lake::ForeignKeyOutcome::Missing { rows } => {
            for row in rows {
                violations.record(con, row, &table.name, &parent.name)?;
            }
            Ok(())
        }
    }
}

pub async fn check_child_fks(
    ctx: &Arc<ExecutionContext>,
    table: &TableEntry,
    batch: &DataBatch,
) -> Result<FkViolations> {
    let mut violations = FkViolations::default();
    let num_rows = batch.columns.first().map(|c| c.len()).unwrap_or(0);
    if num_rows == 0 {
        return Ok(violations);
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

        // A period foreign key matches its last column pair by containment,
        // so it reads the parent's periods rather than asking whether a key
        // exists
        if con.fk_period {
            check_child_period_fk(
                ctx,
                table,
                &parent,
                con,
                batch,
                &local_pos,
                &parent_pos,
                &mut violations,
            )
            .await?;
            continue;
        }

        // A lake parent keeps its keys in data files, not a heap, so the
        // probe reads its manifest statistics instead of a btree or a heap
        // scan. Skipping whole files is what keeps this off the write path
        if parent.lake.is_lake() {
            check_lake_parent(ctx, table, &parent, con, batch, &local_pos, &mut violations)?;
            continue;
        }

        // Single-column FK whose parent column is indexed takes the probe
        // path. Not under a branch: the shared index carries neither the
        // rows the branch appended nor the ones it deleted, so the probe
        // would answer for the main line. The key-set path below reads the
        // branch's view instead, which costs a scan on a branch write and
        // nothing on the main line
        let single_idx = if con.ref_columns.len() == 1 && ctx.active_branch_id.is_none() {
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
                violations.record(con, row, &table.name, &parent.name)?;
            }
        }
    }
    Ok(violations)
}

// ---------------------------------------------------------------------------
// Parent-side enforcement (ON DELETE / ON UPDATE)
// ---------------------------------------------------------------------------

/// Source operator that yields pre-built batches of rows with their
/// locators. Used to feed gathered child rows into a Delete or Update
/// operator so cascades reuse the standard mutation and index-maintenance
/// path. Batches stay homogeneous per storage kind, the DML operators route
/// each batch to the heap or the columnar mutation path as a whole
struct VecSourceOperator {
    batches: std::collections::VecDeque<ExecutionBatch>,
}

impl Operator for VecSourceOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.batches.pop_front()) })
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
            fractional_digits: c.fractional_digits,
        })
        .collect()
}

/// Scans a child table once and gathers every live, visible row `keep`
/// accepts. Heap-resident and columnar-resident rows are gathered into
/// separate locator-tracked batches so the DML operators route each one to
/// its mutation path whole. Returns the batches plus the total match count.
/// A full scan is used because child foreign keys are typically non-unique,
/// so an index probe would miss duplicate references.
///
/// The rows are chosen by a closure rather than a key set because a period
/// foreign key decides row by row: whether a child is still covered depends
/// on which periods its parent has left, not on whether a key is present
async fn gather_children_matching(
    ctx: &Arc<ExecutionContext>,
    child: &TableEntry,
    keep: &mut (dyn FnMut(&DataBatch, usize) -> bool + Send),
) -> Result<(Vec<ExecutionBatch>, usize)> {
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
            if keep(&row, 0) {
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

    let mut matched = 0usize;
    let mut batches: Vec<ExecutionBatch> = Vec::new();
    if !tuple_ids.is_empty() {
        matched += tuple_ids.len();
        let batch = DataBatch::new(builders.into_iter().map(|b| b.finish()).collect());
        batches.push(ExecutionBatch::with_tuple_ids(batch, tuple_ids));
    }

    // Folded child rows live in columnar segments, gathered through the
    // columnar scan so patch overlays and MVCC visibility apply
    if !child.columnar.segments.is_empty() {
        let mut col_builders = create_builders(&schema, 0);
        let mut col_locators: Vec<zyron_common::RowLocator> = Vec::new();
        let mut op = crate::operator::column_scan::ColumnScanOperator::new_for_dml(
            ctx.clone(),
            child.id,
            schema.clone(),
            None,
        )?;
        while let Some(eb) = op.next().await? {
            let Some(locs) = eb.locators.as_ref() else {
                continue;
            };
            for (row, loc) in locs.iter().enumerate() {
                if !loc.is_columnar() {
                    continue;
                }
                if keep(&eb.batch, row) {
                    for (b, col) in col_builders.iter_mut().zip(&eb.batch.columns) {
                        b.push(&col.get_scalar(row));
                    }
                    col_locators.push(*loc);
                }
            }
        }
        if !col_locators.is_empty() {
            matched += col_locators.len();
            let batch = DataBatch::new(col_builders.into_iter().map(|b| b.finish()).collect());
            batches.push(ExecutionBatch::with_locators(batch, col_locators));
        }
    }

    Ok((batches, matched))
}

/// The plain form of the gather above: every child row whose foreign key is
/// one of `target_keys`.
async fn gather_matching_children(
    ctx: &Arc<ExecutionContext>,
    child: &TableEntry,
    fk_positions: &[usize],
    fk_types: &[TypeId],
    target_keys: &HashSet<Vec<u8>>,
) -> Result<(Vec<ExecutionBatch>, usize)> {
    let mut keep = |batch: &DataBatch, row: usize| {
        // MATCH SIMPLE: a row with any null FK component references nothing
        match composite_key_at(batch, row, fk_positions, fk_types) {
            Some(key) => target_keys.contains(&key),
            None => false,
        }
    };
    gather_children_matching(ctx, child, &mut keep).await
}

/// The key at one row, or None when any component is null.
///
/// MATCH SIMPLE leaves a row with a null component unconstrained, so a null
/// anywhere means the row references nothing and no parent write concerns it.
fn composite_key_at(
    batch: &DataBatch,
    row: usize,
    positions: &[usize],
    types: &[TypeId],
) -> Option<Vec<u8>> {
    if positions.iter().any(|&p| batch.columns[p].is_null(row)) {
        return None;
    }
    encode_composite_key(batch, row, positions, types)
}

/// The stored range at one row, or None when the cell is null or holds
/// something that is not a range.
fn period_bytes(batch: &DataBatch, position: usize, row: usize) -> Option<Vec<u8>> {
    let column = batch.columns.get(position)?;
    if column.is_null(row) {
        return None;
    }
    match column.get_scalar(row) {
        ScalarValue::Binary(bytes) => Some(bytes),
        _ => None,
    }
}

/// Splits a period foreign key's columns into the scalar half and the period,
/// which is the last column of each side.
fn split_period_key(positions: &[usize]) -> Option<(&[usize], usize)> {
    let (period, scalar) = positions.split_last()?;
    Some((scalar, *period))
}

/// The child rows a parent write leaves without cover.
///
/// A period foreign key is satisfied by the union of the parent periods that
/// share the child's scalar key, so taking one parent period away only
/// concerns the children it was holding up, and only where no other parent
/// period covers them. Matching on the declared columns instead compares
/// periods for equality and finds nothing, which is how a temporal child gets
/// orphaned by a delete that reported success.
///
/// `new_batch` carries the parent rows as the statement leaves them for an
/// UPDATE and is absent for a DELETE. The surviving periods are what is
/// visible now, less what the statement takes away, plus what it puts back.
/// Removing a period the statement has already applied finds it gone and
/// removes nothing, so this reads the same before and after the parent's own
/// write and both phases of an action get one answer.
#[allow(clippy::too_many_arguments)]
async fn gather_children_losing_cover(
    ctx: &Arc<ExecutionContext>,
    parent: &TableEntry,
    child: &TableEntry,
    con: &ConstraintEntry,
    old_batch: &DataBatch,
    new_batch: Option<&DataBatch>,
    parent_pos: &[usize],
    child_pos: &[usize],
) -> Result<(Vec<ExecutionBatch>, usize)> {
    let (Some((parent_scalar_pos, parent_period_pos)), Some((child_scalar_pos, child_period_pos))) =
        (split_period_key(parent_pos), split_period_key(child_pos))
    else {
        return Ok((Vec::new(), 0));
    };
    if parent.columns[parent_period_pos].type_id != TypeId::Range
        || child.columns[child_period_pos].type_id != TypeId::Range
    {
        return Err(ZyronError::PlanError(format!(
            "foreign key \"{}\" declares PERIOD but its last column pair is not a range",
            con.name
        )));
    }
    let parent_scalar_types: Vec<TypeId> = parent_scalar_pos
        .iter()
        .map(|&p| parent.columns[p].type_id)
        .collect();
    let child_scalar_types: Vec<TypeId> = child_scalar_pos
        .iter()
        .map(|&p| child.columns[p].type_id)
        .collect();

    let mut surviving = collect_visible_key_periods(
        ctx,
        parent,
        parent_scalar_pos,
        &parent_scalar_types,
        parent_period_pos,
    )
    .await?;

    // Keys this statement touches, and the periods it takes from them. A
    // period is removed by value, which is exact: the parent's periods for one
    // key do not overlap, so no two of its rows carry the same one
    let num_rows = old_batch.columns.first().map(|c| c.len()).unwrap_or(0);
    let mut touched: HashSet<Vec<u8>> = HashSet::new();
    for row in 0..num_rows {
        let Some(key) = composite_key_at(old_batch, row, parent_scalar_pos, &parent_scalar_types)
        else {
            continue;
        };
        touched.insert(key.clone());
        let Some(period) = period_bytes(old_batch, parent_period_pos, row) else {
            continue;
        };
        if let Some(periods) = surviving.get_mut(&key)
            && let Some(at) = periods.iter().position(|held| *held == period)
        {
            periods.swap_remove(at);
        }
    }
    if let Some(new_batch) = new_batch {
        let new_rows = new_batch.columns.first().map(|c| c.len()).unwrap_or(0);
        for row in 0..new_rows {
            let Some(key) =
                composite_key_at(new_batch, row, parent_scalar_pos, &parent_scalar_types)
            else {
                continue;
            };
            let Some(period) = period_bytes(new_batch, parent_period_pos, row) else {
                continue;
            };
            let periods = surviving.entry(key).or_default();
            if !periods.iter().any(|held| *held == period) {
                periods.push(period);
            }
        }
    }
    if touched.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let mut keep = |batch: &DataBatch, row: usize| {
        let Some(key) = composite_key_at(batch, row, child_scalar_pos, &child_scalar_types) else {
            return false;
        };
        if !touched.contains(&key) {
            return false;
        }
        // A null period references nothing, so no parent write strands it
        let Some(period) = period_bytes(batch, child_period_pos, row) else {
            return false;
        };
        let left = surviving.get(&key).map(|p| p.as_slice()).unwrap_or(&[]);
        !periods_cover(left, &period)
    };
    gather_children_matching(ctx, child, &mut keep).await
}

/// Drives a sub-operator to completion, discarding its output batches.
async fn drive_to_completion(mut op: Box<dyn Operator>) -> Result<()> {
    while op.next().await?.is_some() {}
    Ok(())
}

/// Enforces ON DELETE actions for every foreign key that references `parent`
/// when the rows in `old_batch` are deleted. Restrict and NoAction reject the
/// delete when matching child rows exist; Cascade deletes them; SetNull
/// rewrites the child foreign-key columns to null; SetDefault writes the
/// child column's default.
pub async fn enforce_parent_delete(
    ctx: &Arc<ExecutionContext>,
    parent: &TableEntry,
    old_batch: &DataBatch,
    phase: FkPhase,
) -> Result<()> {
    // A decision the leader already made is not remade here. Re-running it
    // could reject a row the group has agreed on, which would leave this node
    // holding a different table from every other one
    if ctx.replication_apply {
        return Ok(());
    }

    let referencing = ctx.catalog.referencing_constraints(parent.id);
    if referencing.is_empty() {
        return Ok(());
    }
    let num_rows = old_batch.columns.first().map(|c| c.len()).unwrap_or(0);
    if num_rows == 0 {
        return Ok(());
    }

    for (child, con) in referencing {
        if !phase.runs(con.on_delete, false) {
            continue;
        }
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

        // A period key is answered by coverage, not by equality: the child
        // rows that matter are the ones the deleted periods were holding up
        // and that no remaining parent period covers
        let (batches, matched_count) = if con.fk_period {
            gather_children_losing_cover(
                ctx,
                parent,
                &child,
                &con,
                old_batch,
                None,
                &parent_pos,
                &child_pos,
            )
            .await?
        } else {
            // Keys of the parent rows being deleted (null components reference
            // no child, so they are skipped).
            let mut target_keys = HashSet::new();
            for row in 0..num_rows {
                if let Some(k) = composite_key_at(old_batch, row, &parent_pos, &parent_types) {
                    target_keys.insert(k);
                }
            }
            if target_keys.is_empty() {
                continue;
            }
            gather_matching_children(ctx, &child, &child_pos, &child_types, &target_keys).await?
        };
        if matched_count == 0 {
            continue;
        }

        match con.on_delete {
            ReferentialAction::NoAction | ReferentialAction::Restrict => {
                return Err(ZyronError::ForeignKeyViolation(format!(
                    "delete on table \"{}\" violates foreign key constraint \"{}\" on \"{}\": {} dependent row(s) remain",
                    parent.name, con.name, child.name, matched_count
                )));
            }
            ReferentialAction::Cascade => {
                run_with_depth_guard(async {
                    let source = Box::new(VecSourceOperator {
                        batches: batches.into(),
                    });
                    let del = Box::new(DeleteOperator::new(source, Arc::clone(ctx), child.id));
                    drive_to_completion(del).await
                })
                .await?;
            }
            ReferentialAction::SetNull => {
                run_with_depth_guard(async { drive_set_null(ctx, &child, &con, batches).await })
                    .await?;
            }
            ReferentialAction::SetDefault => {
                run_with_depth_guard(async { drive_set_default(ctx, &child, &con, batches).await })
                    .await?;
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
    batches: Vec<ExecutionBatch>,
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
    drive_child_update(ctx, child, assignments, batches).await
}

/// Updates the gathered child rows, setting each foreign-key column to its
/// declared DEFAULT, or to null where the column declares none.
///
/// The default has to satisfy the constraint it is being written under, so a
/// default naming a parent key that does not exist is refused rather than
/// written: SET DEFAULT that leaves a dangling reference is the violation it
/// was meant to repair. The gathered rows are re-checked through the child's
/// own foreign keys by the update path below.
async fn drive_set_default(
    ctx: &Arc<ExecutionContext>,
    child: &TableEntry,
    con: &ConstraintEntry,
    batches: Vec<ExecutionBatch>,
) -> Result<()> {
    let defaults = zyron_planner::bind_column_defaults(&ctx.catalog, child, &con.columns).await?;
    let assignments: Vec<BoundAssignment> = defaults
        .into_iter()
        .map(|(column_id, value)| BoundAssignment { column_id, value })
        .collect();
    drive_child_update(ctx, child, assignments, batches).await
}

/// Rewrites the gathered child rows' foreign key to the parent's new key.
///
/// A parent update can move many keys at once, and each child follows the
/// one it referenced, so the rows are grouped by the key they hold and each
/// group is written with its own literal. Grouping happens over rows already
/// gathered, so the child is still scanned once however many keys moved.
#[allow(clippy::too_many_arguments)]
async fn drive_cascade_update(
    ctx: &Arc<ExecutionContext>,
    child: &TableEntry,
    con: &ConstraintEntry,
    child_pos: &[usize],
    child_types: &[TypeId],
    assign_columns: &[ColumnId],
    remap: &HashMap<Vec<u8>, Vec<ScalarValue>>,
    batches: Vec<ExecutionBatch>,
) -> Result<()> {
    // The child columns the assignment writes, in the constraint's order, so
    // component i of a new key lands on the column that held component i of
    // the old one. A period key writes only its scalar half, so the caller
    // says which columns move rather than the constraint's whole column list
    let assign_types: Vec<TypeId> = assign_columns
        .iter()
        .filter_map(|cid| child.columns.iter().find(|c| c.id == *cid))
        .map(|c| c.type_id)
        .collect();
    if assign_types.len() != assign_columns.len() {
        return Err(ZyronError::ForeignKeyViolation(format!(
            "constraint \"{}\" names a column table \"{}\" does not have",
            con.name, child.name
        )));
    }

    // Rows grouped by the key they currently hold. A batch keeps its store's
    // locators, so each group is rebuilt as its own batch with the matching
    // locator slice and the update routes it the way it arrived
    let mut groups: HashMap<&Vec<u8>, Vec<ExecutionBatch>> = HashMap::new();
    for eb in &batches {
        let mut masks: HashMap<&Vec<u8>, Vec<bool>> = HashMap::new();
        for row in 0..eb.batch.num_rows {
            let Some(key) = encode_composite_key(&eb.batch, row, child_pos, child_types) else {
                continue;
            };
            let Some((stored, _)) = remap.get_key_value(&key) else {
                continue;
            };
            masks
                .entry(stored)
                .or_insert_with(|| vec![false; eb.batch.num_rows])[row] = true;
        }
        for (key, mask) in masks {
            let batch = eb.batch.filter(&mask);
            if batch.num_rows == 0 {
                continue;
            }
            let grouped = match &eb.locators {
                Some(locs) => ExecutionBatch::with_locators(
                    batch,
                    locs.iter()
                        .zip(mask.iter())
                        .filter(|(_, keep)| **keep)
                        .map(|(loc, _)| *loc)
                        .collect(),
                ),
                None => ExecutionBatch::new(batch),
            };
            groups.entry(key).or_default().push(grouped);
        }
    }

    // The new key travels as parameter values rather than literals. A
    // literal is a parser value, which cannot hold a UUID, a sixteen-byte
    // integer or a binary key exactly, and a key rewritten through its text
    // rendering is a different key
    let assignments: Vec<BoundAssignment> = assign_columns
        .iter()
        .zip(assign_types.iter())
        .enumerate()
        .map(|(index, (cid, type_id))| BoundAssignment {
            column_id: *cid,
            // Parameter indices are one based, matching $1
            value: BoundExpr::Parameter {
                index: index + 1,
                type_id: *type_id,
            },
        })
        .collect();

    for (key, grouped) in groups {
        drive_child_update_with_params(
            ctx,
            child,
            assignments.clone(),
            grouped,
            Some(remap[key].clone()),
        )
        .await?;
    }
    Ok(())
}

/// Applies one assignment set to gathered child rows through the ordinary
/// update path, so the cascaded image passes the child's CHECK constraints,
/// its own foreign keys, its triggers and its index maintenance.
///
/// A column that must stay non-null, or any other constraint the cascaded
/// image breaks, blocks the cascade rather than writing a violating row.
async fn drive_child_update(
    ctx: &Arc<ExecutionContext>,
    child: &TableEntry,
    assignments: Vec<BoundAssignment>,
    batches: Vec<ExecutionBatch>,
) -> Result<()> {
    drive_child_update_with_params(ctx, child, assignments, batches, None).await
}

async fn drive_child_update_with_params(
    ctx: &Arc<ExecutionContext>,
    child: &TableEntry,
    assignments: Vec<BoundAssignment>,
    batches: Vec<ExecutionBatch>,
    params: Option<Vec<ScalarValue>>,
) -> Result<()> {
    let source = Box::new(VecSourceOperator {
        batches: batches.into(),
    });
    let checks = zyron_planner::bind_table_check_constraints(&ctx.catalog, child).await?;
    let generated = zyron_planner::bind_table_generated_columns(&ctx.catalog, child).await?;
    let mut upd = UpdateOperator::new(
        source,
        Arc::clone(ctx),
        child.id,
        assignments,
        logical_schema(child),
        checks,
        generated,
    );
    if let Some(params) = params {
        upd = upd.with_params(params);
    }
    drive_to_completion(Box::new(upd)).await
}

/// When a referential action runs relative to the parent's own write.
///
/// An action that leaves the child pointing at a parent key has to be
/// checked against the parent as the statement leaves it, not as it found
/// it. Running ON UPDATE CASCADE first would point children at a key that
/// does not exist yet; running SET DEFAULT first would let a default that
/// names a row the same statement is about to remove or move pass its own
/// foreign key and then dangle.
///
/// Everything else runs first, so a rejection lands before any row is
/// written: RESTRICT and NO ACTION reject, SET NULL points at nothing, and
/// ON DELETE CASCADE removes the children outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FkPhase {
    BeforeWrite,
    AfterWrite,
}

impl FkPhase {
    /// True when the action leaves the child referencing a parent key, so it
    /// has to see the parent's final state.
    #[inline]
    fn needs_final_parent(action: ReferentialAction, is_update: bool) -> bool {
        match action {
            ReferentialAction::SetDefault => true,
            // A delete cascade removes the child, an update cascade repoints
            // it at the new key
            ReferentialAction::Cascade => is_update,
            _ => false,
        }
    }

    #[inline]
    fn runs(&self, action: ReferentialAction, is_update: bool) -> bool {
        (*self == FkPhase::AfterWrite) == Self::needs_final_parent(action, is_update)
    }
}

/// Enforces ON UPDATE actions when a parent row's referenced key changes.
/// Restrict and NoAction reject the change when dependents exist; SetNull
/// nulls the child key; SetDefault writes the child column's default;
/// Cascade rewrites each child key to the value its parent key moved to.
pub async fn enforce_parent_update(
    ctx: &Arc<ExecutionContext>,
    parent: &TableEntry,
    old_batch: &DataBatch,
    new_batch: &DataBatch,
    phase: FkPhase,
) -> Result<()> {
    // A decision the leader already made is not remade here. Re-running it
    // could reject a row the group has agreed on, which would leave this node
    // holding a different table from every other one
    if ctx.replication_apply {
        return Ok(());
    }

    let referencing = ctx.catalog.referencing_constraints(parent.id);
    if referencing.is_empty() {
        return Ok(());
    }
    let num_rows = old_batch.columns.first().map(|c| c.len()).unwrap_or(0);
    if num_rows == 0 {
        return Ok(());
    }

    for (child, con) in referencing {
        // Skipping before the child scan is what keeps the phase split from
        // costing a second pass over the child table
        if !phase.runs(con.on_update, true) {
            continue;
        }
        let Some(parent_pos) = column_positions(parent, &con.ref_columns) else {
            continue;
        };
        let Some(child_pos) = column_positions(&child, &con.columns) else {
            continue;
        };
        let child_types: Vec<TypeId> = child_pos
            .iter()
            .map(|&p| child.columns[p].type_id)
            .collect();

        // A period key cascades on its scalar half. A child's period is its
        // own, and the parent's new period is not a value that can be written
        // onto it, so the remap and the cascade both work on the columns
        // before the period
        let (key_parent_pos, key_child_pos) = if con.fk_period {
            match (split_period_key(&parent_pos), split_period_key(&child_pos)) {
                (Some((p, _)), Some((c, _))) => (p, c),
                _ => continue,
            }
        } else {
            (&parent_pos[..], &child_pos[..])
        };
        let key_parent_types: Vec<TypeId> = key_parent_pos
            .iter()
            .map(|&p| parent.columns[p].type_id)
            .collect();
        let key_child_types: Vec<TypeId> = key_child_pos
            .iter()
            .map(|&p| child.columns[p].type_id)
            .collect();

        // Only rows whose referenced key actually changed have any effect,
        // and each one carries where its key moved to, so a cascade follows
        // every key of a multi-row update rather than one of them
        let mut remap: HashMap<Vec<u8>, Vec<ScalarValue>> = HashMap::new();
        for row in 0..num_rows {
            let Some(old_key) =
                encode_composite_key(old_batch, row, key_parent_pos, &key_parent_types)
            else {
                continue;
            };
            let new_key = encode_composite_key(new_batch, row, key_parent_pos, &key_parent_types);
            if Some(&old_key) == new_key.as_ref() {
                continue;
            }
            let new_values: Vec<ScalarValue> = key_parent_pos
                .iter()
                .map(|&p| new_batch.columns[p].get_scalar(row))
                .collect();
            // The referenced columns are a key, so one old value cannot move
            // to two places. Two rows claiming the same old key means the
            // key is not unique and the cascade has no single answer
            if let Some(existing) = remap.get(&old_key) {
                if *existing != new_values {
                    return Err(ZyronError::ForeignKeyViolation(format!(
                        "update on table \"{}\" moves the key referenced by constraint \"{}\" to two different values in one statement",
                        parent.name, con.name
                    )));
                }
                continue;
            }
            remap.insert(old_key, new_values);
        }
        // A period key is answered by coverage: the parent's period may have
        // moved without its scalar key moving at all, and the children that
        // matter are the ones no remaining parent period covers
        let (batches, matched_count) = if con.fk_period {
            gather_children_losing_cover(
                ctx,
                parent,
                &child,
                &con,
                old_batch,
                Some(new_batch),
                &parent_pos,
                &child_pos,
            )
            .await?
        } else {
            if remap.is_empty() {
                continue;
            }
            let changed_keys: HashSet<Vec<u8>> = remap.keys().cloned().collect();
            gather_matching_children(ctx, &child, &child_pos, &child_types, &changed_keys).await?
        };
        if matched_count == 0 {
            continue;
        }

        match con.on_update {
            ReferentialAction::NoAction | ReferentialAction::Restrict => {
                return Err(ZyronError::ForeignKeyViolation(format!(
                    "update on table \"{}\" violates foreign key constraint \"{}\" on \"{}\": {} dependent row(s) reference the old key",
                    parent.name, con.name, child.name, matched_count
                )));
            }
            ReferentialAction::SetNull => {
                run_with_depth_guard(async { drive_set_null(ctx, &child, &con, batches).await })
                    .await?;
            }
            ReferentialAction::SetDefault => {
                run_with_depth_guard(async { drive_set_default(ctx, &child, &con, batches).await })
                    .await?;
            }
            ReferentialAction::Cascade => {
                // A cascade moves a child to wherever its parent's key went.
                // A child that lost cover because the parent's period shrank
                // has a key that did not move, so there is nowhere to move it
                // to, and writing nothing would leave an orphan behind a
                // statement that reported success
                if con.fk_period {
                    for eb in &batches {
                        for row in 0..eb.batch.num_rows {
                            let moved =
                                composite_key_at(&eb.batch, row, key_child_pos, &key_child_types)
                                    .is_some_and(|key| remap.contains_key(&key));
                            if !moved {
                                return Err(ZyronError::ForeignKeyViolation(format!(
                                    "update on table \"{}\" leaves a row of \"{}\" outside every period of foreign key constraint \"{}\", and a cascade cannot move a row whose referenced key did not move",
                                    parent.name, child.name, con.name
                                )));
                            }
                        }
                    }
                }
                // A period key moves its scalar half only, so the
                // assignment stops before the period column
                let assign_columns = if con.fk_period {
                    &con.columns[..con.columns.len().saturating_sub(1)]
                } else {
                    &con.columns[..]
                };
                run_with_depth_guard(async {
                    drive_cascade_update(
                        ctx,
                        &child,
                        &con,
                        key_child_pos,
                        &key_child_types,
                        assign_columns,
                        &remap,
                        batches,
                    )
                    .await
                })
                .await?;
            }
        }
    }
    Ok(())
}

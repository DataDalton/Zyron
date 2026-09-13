//! Rewriting a column whose new type cannot read the old bytes.
//!
//! Most type changes are answered by reading what is on disk a wider way, and
//! those are a catalog write. The rest are not: TEXT to INT reinterprets every
//! byte, so the rows have to be re-encoded. That is what runs here, and it runs
//! with the table open the whole time.
//!
//! ## The sequence
//!
//! **Publish** creates a hidden table with the new column type and its own heap
//! files, and registers it in the source's maintenance list. From that instant
//! every insert, update and delete on the source applies to both heaps.
//!
//! **Wait** holds until every transaction that was running at publication has
//! ended, for the same reason an index build waits: a transaction that
//! resolved its maintenance list earlier will not mirror its writes.
//!
//! **Copy** streams the source under one snapshot, casts, and writes into the
//! shadow at background priority.
//!
//! **Catch up** applies what committed after the snapshot but was written by
//! transactions that were already running at publication, which is the only set
//! the copy missed and the hook did not carry. The wait bounds it.
//!
//! **Swap** points the catalog's heap files, columns and epochs at the shadow's
//! in one record. Statements that already opened the old files finish on them.
//!
//! ## What a cast failure does
//!
//! A value the new type cannot hold ends the rewrite: the shadow and its files
//! go, the maintenance entry goes, and the source is untouched. The row is
//! named, because "one row does not convert" is not something an operator can
//! act on. A writer that supplies such a value during the copy fails its own
//! statement instead, and the rewrite carries on.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use zyron_catalog::{ColumnEntry, ShadowSpec, TableEntry};
use zyron_common::{RowLocator, TypeId, ZyronError};
use zyron_executor::column::ScalarValue;

use crate::connection::ServerState;
use crate::ddl_progress::{BuildPacer, DdlOperation, DdlPhase};

/// A rewrite in flight, as this process holds it.
///
/// The catalog carries the same facts in the source's maintenance list, which
/// is what writers read. This is the copy the driver and recovery work from.
pub struct ShadowTarget {
    pub source_table_id: u32,
    pub shadow_table_id: u32,
    pub shadow_heap_file_id: u32,
    pub shadow_fsm_file_id: u32,
    pub column_name: String,
    pub abandoned: Arc<AtomicBool>,
}

/// The name a shadow carries in the catalog.
///
/// Prefixed so nothing can create a user table that collides with one, and
/// suffixed with the source's id so two rewrites cannot pick the same name.
fn shadow_name(source: &TableEntry) -> String {
    format!("zyron_shadow_{}_{}", source.id.0, source.name)
}

/// Whether a table is a shadow rather than something a user created.
///
/// Every listing filters on this, because a shadow exists only between a
/// publish and a swap and a user who saw one would be looking at a table that
/// vanishes.
pub fn is_shadow_table(name: &str) -> bool {
    name.starts_with("zyron_shadow_")
}

/// What one attempt produced.
pub enum RewriteOutcome {
    /// The swap installed the rewritten table
    Swapped { rows: u64 },
    /// A value the new type cannot hold ended it, and nothing changed
    CastFailed {
        locator: RowLocator,
        value: String,
        reason: String,
    },
}

/// Runs an incompatible SET TYPE with the table open throughout.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    source: &Arc<TableEntry>,
    column_name: &str,
    target_type: TypeId,
    target_digits: Option<u8>,
    target_max_length: Option<usize>,
    issuing_session: &str,
    hold_open: &[u64],
) -> Result<RewriteOutcome, ZyronError> {
    let column = source
        .live_columns()
        .find(|c| c.name.eq_ignore_ascii_case(column_name))
        .ok_or_else(|| ZyronError::ColumnNotFound(column_name.to_string()))?
        .clone();

    let progress = server.ddl_progress.begin(
        &source.name,
        &column.name,
        DdlOperation::ShadowRewrite,
        issuing_session,
    );
    progress.set_phase(DdlPhase::Publishing);

    // ---- Publish -------------------------------------------------------
    let (shadow_heap_id, shadow_fsm_id) = server.catalog.alloc_heap_files();
    let shadow_name = shadow_name(source);
    let shadow_id = create_shadow_table(
        server,
        schema_id,
        source,
        &shadow_name,
        &column,
        target_type,
        target_digits,
        target_max_length,
        shadow_heap_id,
        shadow_fsm_id,
    )
    .await?;

    // The shadow's heap is opened here, with no log attached, before any
    // writer can be told to mirror into it, so the instance the writers find
    // in the cache is the unlogged one the fill runs through
    let shadow_entry = server.catalog.get_table_by_id(shadow_id)?;
    crate::index_build::shadow_heap_file(server, &shadow_entry).await?;

    let abandoned = Arc::new(AtomicBool::new(false));
    let rows_map = Arc::new(scc::HashMap::new());
    let spec = ShadowSpec {
        shadow_table_id: shadow_id,
        column_id: column.id,
        target_type,
        target_digits,
        target_max_length,
        rows: Arc::clone(&rows_map),
        abandoned: Arc::clone(&abandoned),
    };
    // Recorded before the maintenance list, so a crash between the two leaves
    // a target recovery can find and clean up rather than an orphan heap
    let _ = server
        .shadow_targets
        .insert_async(
            source.id.0,
            Arc::new(ShadowTarget {
                source_table_id: source.id.0,
                shadow_table_id: shadow_id.0,
                shadow_heap_file_id: shadow_heap_id,
                shadow_fsm_file_id: shadow_fsm_id,
                column_name: column.name.clone(),
                abandoned: Arc::clone(&abandoned),
            }),
        )
        .await;
    let active_at_publication = server.txn_manager.proc_array().active_txn_ids();
    server
        .catalog
        .set_shadow_targets(source.id, vec![spec.clone()]);

    // Everything after this point undoes the publish on the way out
    let result = drive(
        server,
        schema_id,
        source,
        &column,
        &spec,
        shadow_id,
        &shadow_name,
        &active_at_publication,
        hold_open,
        progress.progress(),
    )
    .await;

    match result {
        Ok(RewriteOutcome::Swapped { rows }) => {
            server.catalog.set_shadow_targets(source.id, Vec::new());
            let _ = server.shadow_targets.remove_async(&source.id.0).await;
            Ok(RewriteOutcome::Swapped { rows })
        }
        Ok(other) => {
            abandoned.store(true, Ordering::Release);
            discard_shadow(server, source.id, shadow_id, shadow_heap_id, shadow_fsm_id).await;
            Ok(other)
        }
        Err(e) => {
            abandoned.store(true, Ordering::Release);
            discard_shadow(server, source.id, shadow_id, shadow_heap_id, shadow_fsm_id).await;
            Err(e)
        }
    }
}

/// Removes a shadow and everything that pointed at it.
///
/// The maintenance entry goes first, so no writer is still mirroring into a
/// table whose files are being deleted.
pub async fn discard_shadow(
    server: &Arc<ServerState>,
    source_id: zyron_catalog::TableId,
    shadow_id: zyron_catalog::TableId,
    heap_file_id: u32,
    fsm_file_id: u32,
) {
    server.catalog.set_shadow_targets(source_id, Vec::new());
    let _ = server.shadow_targets.remove_async(&source_id.0).await;
    if let Ok(entry) = server.catalog.get_table_by_id(shadow_id) {
        let _ = server
            .catalog
            .drop_table(entry.schema_id, &entry.name)
            .await;
    }
    let _ = server.heap_files.remove_async(&heap_file_id).await;
    let _ = server.disk_manager.delete_file(heap_file_id).await;
    let _ = server.disk_manager.delete_file(fsm_file_id).await;
}

/// Creates the hidden table the rows are copied into.
#[allow(clippy::too_many_arguments)]
async fn create_shadow_table(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    source: &Arc<TableEntry>,
    name: &str,
    column: &ColumnEntry,
    target_type: TypeId,
    target_digits: Option<u8>,
    target_max_length: Option<usize>,
    heap_file_id: u32,
    fsm_file_id: u32,
) -> Result<zyron_catalog::TableId, ZyronError> {
    // The shadow holds only the columns a user can see: a dropped column's
    // placeholder is exactly what a rewrite is free to leave behind
    let mut columns: Vec<ColumnEntry> = source.live_column_list();
    for (i, c) in columns.iter_mut().enumerate() {
        c.ordinal = i as u16;
        c.absent_value = None;
        c.dropped = false;
        if c.id == column.id {
            c.type_id = target_type;
            c.fractional_digits = target_digits;
            c.max_length = target_max_length;
        }
    }
    server
        .catalog
        .create_table_with_files(
            schema_id,
            name,
            columns,
            Vec::new(),
            heap_file_id,
            fsm_file_id,
        )
        .await
}

/// The copy, catch-up, index build and swap, with the publish already done.
#[allow(clippy::too_many_arguments)]
async fn drive(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    source: &Arc<TableEntry>,
    column: &ColumnEntry,
    spec: &ShadowSpec,
    shadow_id: zyron_catalog::TableId,
    shadow_name: &str,
    active_at_publication: &[u64],
    hold_open: &[u64],
    progress: &crate::ddl_progress::DdlProgress,
) -> Result<RewriteOutcome, ZyronError> {
    // ---- Wait ----------------------------------------------------------
    progress.set_phase(DdlPhase::WaitingOldTxns);
    crate::index_build::wait_for_transactions_active_at(
        server,
        active_at_publication,
        hold_open,
        None,
    )
    .await?;

    // ---- Copy ----------------------------------------------------------
    progress.set_phase(DdlPhase::Scanning);
    let mut pacer = BuildPacer::new(DdlPhase::Scanning);
    let mut stream = crate::index_build::LiveRowStream::open(
        server,
        source,
        crate::index_build::DEFAULT_BUILD_BATCH_ROWS,
    )
    .await?;
    progress.set_rows_total_estimate(stream.rows_total_estimate());

    let position = source
        .columns
        .iter()
        .position(|c| c.id == column.id)
        .ok_or_else(|| ZyronError::ColumnNotFound(column.name.clone()))?;
    let shadow_entry = server.catalog.get_table_by_id(shadow_id)?;
    let mut copied: u64 = 0;

    while let Some(batch) = stream.next_batch().await? {
        if batch.is_empty() {
            continue;
        }
        let (decoded, locators) = crate::index_build::decode_live_batch(source, &batch)?;
        match cast_batch(&decoded, position, spec) {
            Ok(column) => {
                // The batch the decode produced is this loop's own, so the cast
                // column takes the original's place rather than being carried
                // into a second copy of every column beside it
                let mut casted = decoded;
                casted.columns[position] = column;
                copied += casted.num_rows as u64;
                write_shadow_batch(server, &shadow_entry, &casted, &locators, spec).await?;
                progress.add_rows(casted.num_rows as u64);
            }
            Err((row, reason)) => {
                let value = describe_value(&decoded, position, row);
                let locator = locators.get(row).copied().unwrap_or(RowLocator::Heap {
                    page: zyron_common::page::PageId::new(source.heap_file_id, 0),
                    slot: 0,
                });
                return Ok(RewriteOutcome::CastFailed {
                    locator,
                    value,
                    reason,
                });
            }
        }
        pacer.between_batches(progress).await;
    }
    stream.close();

    // ---- Catch up ------------------------------------------------------
    progress.set_phase(DdlPhase::CatchingUp);
    // The copy read one snapshot. Rows a transaction that was already running
    // at publication committed after that snapshot are the only ones the copy
    // missed and the hook did not carry, because the hook only covers writers
    // that resolved their maintenance list after publication. The wait above
    // is what bounds that set: every such transaction had already ended, so
    // the set is what they committed between the wait and the snapshot
    let caught = catch_up_from_source(server, source, &shadow_entry, position, spec).await?;
    copied += caught;

    // ---- Indexes -------------------------------------------------------
    // Built before the swap so the swap installs a table that answers every
    // access path the source did
    build_shadow_indexes(server, schema_id, source, shadow_name, shadow_id, hold_open).await?;

    // ---- Swap ----------------------------------------------------------
    progress.set_phase(DdlPhase::Swapping);
    swap_in_shadow(server, source, &shadow_entry, column, spec).await?;
    Ok(RewriteOutcome::Swapped { rows: copied })
}

/// Casts the changed column of one batch, naming the first row that will not
/// convert.
///
/// Only the one column comes back. The caller puts it in place of the original
/// in the batch it already owns, so a copy of every other column is not made
/// on the way past.
fn cast_batch(
    batch: &zyron_executor::batch::DataBatch,
    position: usize,
    spec: &ShadowSpec,
) -> Result<zyron_executor::column::Column, (usize, String)> {
    let source = &batch.columns[position];
    let casted = if spec.target_type == TypeId::Decimal {
        zyron_executor::compute::cast_column_to_decimal(source, spec.target_digits.unwrap_or(0))
    } else {
        zyron_executor::compute::cast_column(source, spec.target_type)
    };
    match casted {
        Ok(column) => Ok(column),
        Err(e) => {
            // The column cast reports the failure without saying which row, so
            // the row is found by casting one value at a time. Only a failing
            // batch pays for this
            for row in 0..batch.num_rows {
                let mut single = zyron_executor::batch::ColumnBuilder::new(source.type_id, 1);
                single.push(&source.get_scalar(row));
                let one = single.finish();
                let attempt = if spec.target_type == TypeId::Decimal {
                    zyron_executor::compute::cast_column_to_decimal(
                        &one,
                        spec.target_digits.unwrap_or(0),
                    )
                } else {
                    zyron_executor::compute::cast_column(&one, spec.target_type)
                };
                if attempt.is_err() {
                    return Err((row, e.to_string()));
                }
            }
            Err((0, e.to_string()))
        }
    }
}

/// Renders one value for the failure report, so the operator can find the row
/// by its content rather than only by its address.
fn describe_value(batch: &zyron_executor::batch::DataBatch, position: usize, row: usize) -> String {
    match batch.columns[position].get_scalar(row) {
        ScalarValue::Utf8(s) => s,
        ScalarValue::Null => "NULL".to_string(),
        other => format!("{other:?}"),
    }
}

/// Writes one cast batch into the shadow heap and records where each row
/// landed.
async fn write_shadow_batch(
    server: &Arc<ServerState>,
    shadow: &Arc<TableEntry>,
    batch: &zyron_executor::batch::DataBatch,
    source_locators: &[RowLocator],
    spec: &ShadowSpec,
) -> Result<(), ZyronError> {
    let tuples = zyron_executor::batch::batch_to_tuples(
        batch,
        &shadow.columns,
        // Rows the copy writes are visible to every reader of the swapped
        // table, so they carry the frozen id rather than a transaction that
        // could still abort
        1,
        shadow.schema_epoch,
    );
    let heap = crate::index_build::shadow_heap_file(server, shadow).await?;
    let written = heap.insert_batch(&tuples).await?;
    // One entry per row, taken without an await apiece. The map is not
    // contended for these keys, because no other writer holds a source address
    // this copy is placing, so the synchronous path takes the bucket lock and
    // returns rather than suspending the copy once per row
    for (i, shadow_id) in written.iter().enumerate() {
        let Some(source) = source_locators.get(i) else {
            continue;
        };
        let Some(key) = zyron_executor::shadow_write::pack_locator(*source) else {
            continue;
        };
        let value = (shadow_id.page_id.page_num << 16) | shadow_id.slot_id as u64;
        let _ = spec.rows.insert_sync(key, value);
    }
    Ok(())
}

/// Applies rows the copy's snapshot did not see.
///
/// The copy read one snapshot of the source. A row a transaction committed
/// after that snapshot is either mirrored by the hook, which covers every
/// writer that resolved its maintenance list after publication, or written by
/// one of the transactions the wait already drained. Reading the source again
/// and taking every row the row map does not already name covers the second
/// set exactly.
async fn catch_up_from_source(
    server: &Arc<ServerState>,
    source: &Arc<TableEntry>,
    shadow: &Arc<TableEntry>,
    position: usize,
    spec: &ShadowSpec,
) -> Result<u64, ZyronError> {
    let mut stream = crate::index_build::LiveRowStream::open(
        server,
        source,
        crate::index_build::DEFAULT_BUILD_BATCH_ROWS,
    )
    .await?;
    let mut applied: u64 = 0;
    while let Some(batch) = stream.next_batch().await? {
        if batch.is_empty() {
            continue;
        }
        let (decoded, locators) = crate::index_build::decode_live_batch(source, &batch)?;
        let mut keep = vec![false; locators.len()];
        let mut any = false;
        for (i, locator) in locators.iter().enumerate() {
            let Some(key) = zyron_executor::shadow_write::pack_locator(*locator) else {
                continue;
            };
            if spec.rows.read_async(&key, |_, _| ()).await.is_none() {
                keep[i] = true;
                any = true;
            }
        }
        if !any {
            continue;
        }
        let picked = zyron_executor::batch::DataBatch::new(
            decoded.columns.iter().map(|c| c.filter(&keep)).collect(),
        );
        let picked_locators: Vec<RowLocator> = locators
            .iter()
            .zip(keep.iter())
            .filter(|(_, k)| **k)
            .map(|(l, _)| *l)
            .collect();
        let column = cast_batch(&picked, position, spec).map_err(|(row, reason)| {
            ZyronError::ExecutionError(format!(
                "a row written during the rewrite does not convert: row {row}, {reason}"
            ))
        })?;
        let mut casted = picked;
        casted.columns[position] = column;
        applied += casted.num_rows as u64;
        write_shadow_batch(server, shadow, &casted, &picked_locators, spec).await?;
    }
    stream.close();
    Ok(applied)
}

/// Builds every index the source declares on the shadow, so the swap installs
/// a fully indexed table.
async fn build_shadow_indexes(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    source: &Arc<TableEntry>,
    shadow_name: &str,
    shadow_id: zyron_catalog::TableId,
    hold_open: &[u64],
) -> Result<(), ZyronError> {
    let indexes = server.catalog.get_indexes_for_table(source.id);
    for index in indexes {
        if index.index_type != zyron_catalog::IndexType::BTree {
            continue;
        }
        let columns: Vec<(String, bool)> = index
            .columns
            .iter()
            .filter_map(|ic| {
                source
                    .columns
                    .iter()
                    .find(|c| c.id == ic.column_id)
                    .map(|c| (c.name.clone(), ic.descending))
            })
            .collect();
        if columns.len() != index.columns.len() {
            continue;
        }
        let name = format!("{shadow_name}_{}", index.name);
        crate::ddl_dispatch::build_heap_btree_index(
            server,
            schema_id,
            shadow_id,
            &name,
            &columns,
            index.unique,
            "shadow rewrite",
            hold_open,
        )
        .await?;
    }
    Ok(())
}

/// Points the source's catalog entry at the shadow's storage in one record.
///
/// Statements that already opened the old files finish on them, because a
/// heap file handle outlives the entry that named it. New statements resolve
/// the entry and open the new files.
async fn swap_in_shadow(
    server: &Arc<ServerState>,
    source: &Arc<TableEntry>,
    shadow: &Arc<TableEntry>,
    column: &ColumnEntry,
    spec: &ShadowSpec,
) -> Result<(), ZyronError> {
    let old_heap = source.heap_file_id;
    let old_fsm = source.fsm_file_id;

    // The fill ran with no log attached. From here every page change is
    // recorded, and every page the fill left dirty goes to disk and is
    // synced, so once the catalog names these files as the table's each row
    // is either on disk or in the log. The log is attached first, so a row
    // a writer mirrors while the flush runs is recorded rather than left on
    // a page the flush already copied
    let heap = crate::index_build::shadow_heap_file(server, shadow).await?;
    heap.attach_wal(&server.wal);
    heap.flush().await?;
    server.disk_manager.fsync_file(shadow.heap_file_id)?;
    server.disk_manager.fsync_file(shadow.fsm_file_id)?;

    let mut entry = server.catalog.get_table_by_id(source.id)?.as_ref().clone();
    entry.heap_file_id = shadow.heap_file_id;
    entry.fsm_file_id = shadow.fsm_file_id;
    entry.columns = shadow.columns.clone();
    for c in entry.columns.iter_mut() {
        c.table_id = source.id;
    }
    // The rewritten heap holds one layout, so the table's epoch history starts
    // over: nothing on disk was written under any earlier one
    entry.seal_initial_epoch();
    // The rewrite re-encoded every row, so no folded segment describes the
    // table any more
    entry.columnar = Default::default();
    entry
        .constraints
        .retain(|c| !c.columns.contains(&column.id));
    server.catalog.update_table(entry).await?;

    // The table's rows are the shadow's rows from here, so a write mirrored
    // into the shadow would land in the file it just wrote. Clearing the
    // targets before the shadow's catalog row goes is what stops a writer
    // resolving a spec that names a table no longer there
    debug_assert_eq!(
        spec.shadow_table_id, shadow.id,
        "the swap is clearing a spec for a different shadow"
    );
    server.catalog.set_shadow_targets(source.id, Vec::new());

    // The shadow's catalog row has done its job. Its files stay, because they
    // are the table's files now
    let _ = server
        .catalog
        .drop_table(shadow.schema_id, &shadow.name)
        .await;

    // Old files are released once no statement can still be reading them,
    // which the heap file cache decides by dropping the last handle
    let _ = server.heap_files.remove_async(&old_heap).await;
    if let Err(e) = server.disk_manager.delete_file(old_heap).await {
        tracing::warn!(
            target: "zyron::ddl",
            file_id = old_heap,
            error = %e,
            "the rewritten table's old heap file is still held, so it stays on disk until the \
             statements reading it finish"
        );
    }
    if let Err(e) = server.disk_manager.delete_file(old_fsm).await {
        tracing::warn!(
            target: "zyron::ddl",
            file_id = old_fsm,
            error = %e,
            "the rewritten table's old free space map is still held"
        );
    }
    Ok(())
}

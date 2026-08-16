//! Building an index over rows a table already holds.
//!
//! Two statements do it: CREATE INDEX on a populated table, and REINDEX.
//! Both need the same answer to the same question, which rows are live right
//! now, and getting that answer wrong in either direction is a wrong-answer
//! bug rather than a slow one: a missed row is a row every index scan silently
//! drops, and a dead row indexed is a row an index scan resurrects.
//!
//! The visibility rules and the columnar traversal live here once, so the two
//! callers cannot drift apart. Each caller keeps its own insertion loop, since
//! they differ in how they report progress and what they do with the tree.
//!
//! A lake table takes a different path entirely. Its rows are addressed by
//! data file and ordinal rather than by page and slot, and the files a
//! clustering or compaction pass rewrites would invalidate every entry of a
//! tree built over them. Its index is committed into its own transaction log
//! instead, so it is versioned with the rows, maintained by the same commits
//! that move them, and readable at a past version. `build_lake_index` routes
//! there.

use std::sync::Arc;

use zyron_catalog::TableEntry;
use zyron_common::ZyronError;
use zyron_common::page::PageId;
use zyron_executor::context::ExecutionContext;
use zyron_storage::{HeapFile, HeapFileConfig, HeapPage, TupleHeader};

use crate::connection::ServerState;

/// Every row of one table an index must cover, split by where it lives.
///
/// Collected once per table and reused across its indexes, because the rows
/// are the same for all of them and reading the heap once per index would make
/// REINDEX of a table with several indexes cost several full scans.
pub struct LiveRows {
    /// Heap resident rows as (page, [(slot, row image)]), grouped by page so
    /// the index key can be stamped with the row's page and slot.
    pub heap: Vec<(PageId, Vec<(u16, Vec<u8>)>)>,
    /// Folded rows as decoded batches paired with their locators, with the
    /// patch overlay and MVCC visibility already applied. Empty for a table
    /// with no columnar segments.
    pub columnar: Vec<(
        zyron_executor::batch::DataBatch,
        Vec<zyron_common::RowLocator>,
    )>,
}

impl LiveRows {
    /// Upper bound on the entries an index built from these rows will hold,
    /// used to size progress reporting rather than to allocate.
    pub fn row_count(&self) -> u64 {
        let heap: u64 = self.heap.iter().map(|(_, live)| live.len() as u64).sum();
        let columnar: u64 = self.columnar.iter().map(|(b, _)| b.num_rows as u64).sum();
        heap + columnar
    }
}

/// Resolves index column names against a table and returns their ids in the
/// order they were declared.
fn column_ids_for(table: &TableEntry, column_names: &[String]) -> Result<Vec<u32>, ZyronError> {
    let mut ids = Vec::with_capacity(column_names.len());
    for name in column_names {
        let column = table
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                ZyronError::PlanError(format!(
                    "column \"{}\" is not in table \"{}\"",
                    name, table.name
                ))
            })?;
        ids.push(column.id.0 as u32);
    }
    Ok(ids)
}

/// Declares an index on a lake table and backfills it over every live data
/// file in one commit.
///
/// The declaration and its entries land in the same version, so no version
/// exists where the index is declared but empty. Later commits that add or
/// rewrite data files carry their own index entries, and a commit that could
/// not maintain them leaves the index short of covering the table, which makes
/// a probe decline rather than answer with fewer rows than the table has
pub async fn build_lake_index(
    server: &Arc<ServerState>,
    table: &TableEntry,
    column_names: &[String],
    unique: bool,
) -> Result<(), ZyronError> {
    let column_ids = column_ids_for(table, column_names)?;
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
        ZyronError::ConfigError(format!(
            "this node does not run the lake tier, so it cannot index \"{}\"",
            table.name
        ))
    })?;
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SchemaChange,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
    };
    let name = index_name_for(table, column_names);
    let outcome = zyron_lake::operations::create_index(
        &log,
        attempt,
        table.id.0 as u64,
        &name,
        &column_ids,
        unique,
    )?;
    tracing::info!(
        target: "zyron::ddl",
        index = %name,
        table = %table.name,
        version = outcome.version,
        rows = outcome.rows,
        "CREATE INDEX built a lake index"
    );
    Ok(())
}

/// Drops a lake table's index, by the name the build gave it.
///
/// A missing index is not an error here: the catalog entry is the authority
/// on what exists, and this call only removes the storage behind it
pub async fn drop_lake_index(
    server: &Arc<ServerState>,
    table: &TableEntry,
    column_names: &[String],
) -> Result<(), ZyronError> {
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
    let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) else {
        return Ok(());
    };
    let name = index_name_for(table, column_names);
    if log
        .latest_manifest()
        .map(|m| m.index_by_name(&name).is_none())
        .unwrap_or(true)
    {
        return Ok(());
    }
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SchemaChange,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
    };
    zyron_lake::operations::drop_index(&log, attempt, &name)?;
    Ok(())
}

/// Rebuilds every index a lake table declares, which is what REINDEX runs.
pub async fn rebuild_lake_indexes(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Result<(), ZyronError> {
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
    let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) else {
        return Ok(());
    };
    if log
        .latest_manifest()
        .map(|m| m.indexes.is_empty())
        .unwrap_or(true)
    {
        return Ok(());
    }
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SchemaChange,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
    };
    zyron_lake::operations::rebuild_indexes(&log, attempt, table.id.0 as u64)?;
    Ok(())
}

/// The name a lake index carries in its own manifest.
///
/// The catalog's index name is unique per schema while the lake manifest is
/// per table, so the table's own columns are what identify it there. Deriving
/// it from the columns rather than storing it keeps the two from drifting when
/// an index is renamed in the catalog
fn index_name_for(table: &TableEntry, column_names: &[String]) -> String {
    let mut resolved: Vec<String> = Vec::with_capacity(column_names.len());
    for name in column_names {
        match table
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
        {
            Some(column) => resolved.push(column.name.clone()),
            None => resolved.push(name.clone()),
        }
    }
    format!("ix_{}", resolved.join("_"))
}

/// Collects every live row of a table, heap resident and folded alike.
///
/// A heap row is live when it is not reclaimable under the same commit-status
/// and retention rules vacuum applies, so an index built from this set matches
/// the heap's live-plus-retained rows exactly. Folded rows come back through
/// the columnar scan, which applies the patch overlay and snapshot visibility,
/// so a table whose rows have folded is covered rather than silently skipped.
pub async fn collect_live_rows(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Result<LiveRows, ZyronError> {
    let status_map = server.txn_manager.status_map().clone();
    let active_txns = server.txn_manager.active_txn_ids();
    let oldest_active = if active_txns.is_empty() {
        server.txn_manager.next_txn_id()
    } else {
        active_txns[0]
    };
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    let retention_floor = zyron_executor::operator::modify::effective_retention_floor(
        table,
        &status_map,
        server.txn_manager.retention_clock(),
        now_us,
    );
    let is_dead = |xmin: u32, x: u32| {
        status_map.is_aborted(xmin as u64)
            || (x != 0
                && status_map.is_committed(x as u64)
                && (x as u64) < oldest_active
                && status_map.is_reclaimable_below(x as u64, retention_floor))
    };

    // A lake table has no heap file, and opening the handle would materialize
    // an empty one on disk
    let heap = if table.lake.is_lake() {
        Vec::new()
    } else {
        let heap_file = open_heap_file(server, table).await?;
        // Every page of the heap by number, not a scan guard's pinned set.
        // The loop below reads each page from the pool when resident and
        // straight off disk otherwise, so it needs the full extent
        let page_ids: Vec<PageId> = (0..heap_file.num_pages_cached() as u64)
            .map(|n| PageId::new(heap_file.heap_file_id(), n))
            .collect();

        let mut byPage: Vec<(PageId, Vec<(u16, Vec<u8>)>)> = Vec::with_capacity(page_ids.len());
        for page_id in &page_ids {
            let live = match server.buffer_pool.fetch_page(*page_id) {
                Some(frame) => {
                    let guard = frame.read_data();
                    let live = live_rows_in_page(&guard[..], &is_dead);
                    drop(guard);
                    server.buffer_pool.unpin_page(*page_id, false);
                    live
                }
                // A page the pool does not hold is read straight off disk
                // rather than skipped. Skipping it would silently drop every
                // row on it from the index, and a table large enough to want
                // an index is exactly the one whose pages do not all fit.
                // The read bypasses the pool so rebuilding an index over a
                // large table does not evict the serving working set
                None => {
                    let data = server.disk_manager.read_page(*page_id).await?;
                    live_rows_in_page(&data[..], &is_dead)
                }
            };
            if !live.is_empty() {
                byPage.push((*page_id, live));
            }
        }
        byPage
    };

    let columnar = if table.columnar.segments.is_empty() {
        Vec::new()
    } else {
        collect_columnar_rows(server, table).await?
    };

    Ok(LiveRows { heap, columnar })
}

/// Live rows on one heap page, as (slot, row image).
fn live_rows_in_page(data: &[u8], is_dead: &impl Fn(u32, u32) -> bool) -> Vec<(u16, Vec<u8>)> {
    let header = HeapPage::heap_header_from_slice(data);
    let mut live = Vec::new();
    for slot in 0..header.slot_count {
        let Some(view) = HeapPage::get_tuple_view_from_slice(data, zyron_storage::SlotId(slot))
        else {
            continue;
        };
        let hdr: TupleHeader = view.header;
        if is_dead(hdr.xmin, hdr.xmax) {
            continue;
        }
        live.push((slot, view.data.to_vec()));
    }
    live
}

/// Resolves a table's heap file through the server's cache.
///
/// The cached instance is the one the writers used, so its page count includes
/// pages that exist only as dirty frames in the buffer pool. A freshly
/// constructed handle reads its page count off the file on disk and would see
/// none of them, which would report a just-populated table as empty.
async fn open_heap_file(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Result<Arc<HeapFile>, ZyronError> {
    if let Some(hit) = server.heap_files.get_async(&table.heap_file_id).await {
        return Ok(Arc::clone(hit.get()));
    }
    let heap_file = HeapFile::new(
        Arc::clone(&server.disk_manager),
        Arc::clone(&server.buffer_pool),
        HeapFileConfig {
            heap_file_id: table.heap_file_id,
            fsm_file_id: table.fsm_file_id,
        },
    )?;
    heap_file.init_cache().await?;
    let arc = Arc::new(heap_file);
    // A lost race drops this instance and converges on the winner, the same
    // way the execution context's heap file cache resolves one
    match server
        .heap_files
        .insert_async(table.heap_file_id, Arc::clone(&arc))
        .await
    {
        Ok(()) => Ok(arc),
        Err(_) => {
            let hit = server
                .heap_files
                .get_async(&table.heap_file_id)
                .await
                .ok_or_else(|| {
                    ZyronError::Internal(format!(
                        "heap file {} vanished from the cache during an index build",
                        table.heap_file_id
                    ))
                })?;
            Ok(Arc::clone(hit.get()))
        }
    }
}

/// Reads a table's folded rows through the columnar scan, so the patch overlay
/// and MVCC visibility decide what is live rather than a second copy of those
/// rules written here.
async fn collect_columnar_rows(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Result<
    Vec<(
        zyron_executor::batch::DataBatch,
        Vec<zyron_common::RowLocator>,
    )>,
    ZyronError,
> {
    use zyron_executor::operator::Operator;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)?;
    let scan_ctx = Arc::new(ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id as u32,
        txn.snapshot.clone(),
    ));
    let logical: Vec<zyron_planner::logical::LogicalColumn> = table
        .columns
        .iter()
        .map(|c| zyron_planner::logical::LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();

    let scanned = async {
        let mut op = zyron_executor::operator::column_scan::ColumnScanOperator::new_for_dml(
            Arc::clone(&scan_ctx),
            table.id,
            logical,
            None,
        )?;
        let mut out = Vec::new();
        while let Some(eb) = op.next().await? {
            let Some(locs) = eb.locators.clone() else {
                continue;
            };
            out.push((eb.batch, locs));
        }
        Ok::<_, ZyronError>(out)
    }
    .await;

    // The scan is read only, so the transaction has nothing to commit. Aborting
    // regardless of the scan's outcome keeps it from leaking into the active set
    let _ = server.txn_manager.abort(&mut txn);
    scanned
}

/// Fills a B+tree with one index key per live row, keyed on every column of
/// the index in key order.
///
/// Returns the number of entries inserted. A row whose key columns are absent
/// from the table contributes nothing, which is what the underlying rebuild
/// helpers already decide.
pub fn fill_btree_from_live_rows(
    table: &TableEntry,
    rows: &LiveRows,
    key_columns: &[zyron_catalog::ColumnId],
    btree: &Arc<zyron_storage::BTreeIndex>,
) -> u64 {
    let mut inserted: u64 = 0;
    for (page_id, live) in &rows.heap {
        inserted += zyron_executor::operator::modify::rebuild_btree_index_from_rows(
            table,
            *page_id,
            live,
            key_columns,
            btree,
        ) as u64;
    }
    for (batch, locs) in &rows.columnar {
        inserted += zyron_executor::operator::modify::rebuild_btree_index_from_batch(
            table,
            batch,
            locs,
            key_columns,
            btree,
        ) as u64;
    }
    inserted
}

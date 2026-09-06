//! Sequential, parallel, and index scan operators.
//!
//! SeqScanOperator reads heap pages one at a time, decodes visible tuples
//! into columnar batches, and optionally applies a predicate filter.
//! ParallelSeqScanOperator splits the page range across multiple tokio tasks
//! for multi-core throughput on large tables.
//! IndexScanOperator uses B+ tree range scans to collect matching TupleIds,
//! then fetches only those tuples from the heap. Falls back to sequential
//! scan with predicate filtering when no B+ tree instance is registered.

use std::sync::Arc;

use zyron_buffer::BufferPool;
use zyron_catalog::{IndexEntry, TableEntry};
use zyron_common::Result;
use zyron_common::TypeId;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_parser::ast::{BinaryOperator, LiteralValue};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{BTreeIndex, DiskManager, HeapPage, TupleId};

use crate::batch::{
    DataBatch, build_column_to_builder_map, decode_tuple_into_builders, finalize_builders,
};
use crate::column::ScalarValue;
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Reads a heap page through the buffer pool: serves the buffer-pool copy
/// when present (and freshly inserted pages live there before the background
/// writer flushes them), otherwise loads from disk into the pool and returns
/// the loaded data. Any dirty page evicted during the load is written back.
pub(crate) async fn read_page_through_pool(
    pool: &BufferPool,
    disk: &DiskManager,
    page_id: PageId,
) -> Result<[u8; PAGE_SIZE]> {
    if let Some(frame) = pool.fetch_page(page_id) {
        let guard = frame.read_data();
        let data: [u8; PAGE_SIZE] = **guard;
        drop(guard);
        pool.unpin_page(page_id, false);
        return Ok(data);
    }
    let disk_data = disk.read_page(page_id).await?;
    // A pool with no frame to spare is a capacity limit, not a failed read.
    // The page's bytes are already in hand, so the caller gets them and the
    // page simply goes uncached, which is what the heap's own whole-file scan
    // does with the same refusal. Failing here instead would turn a small
    // pool into a query that cannot run
    let frame = match pool.load_page(page_id, &disk_data) {
        Ok(loaded) => loaded,
        Err(zyron_common::ZyronError::BufferPoolFull) => return Ok(disk_data),
        Err(e) => return Err(e),
    };
    let guard = frame.read_data();
    let data: [u8; PAGE_SIZE] = **guard;
    drop(guard);
    pool.unpin_page(page_id, false);
    Ok(data)
}

/// Builders for a heap scan's output columns. An ENCRYPTED column gets a
/// binary container because its stored cells are ciphertext, which the
/// tuple decoder hands over as bytes. The scan-side decrypt then swaps in
/// the logical text column before anything reads the batch. A text
/// container would silently drop every ciphertext cell on the type
/// mismatch and the decrypt would have nothing to decrypt
pub(crate) fn scan_builders(
    output_columns: &[LogicalColumn],
    table_columns: &[zyron_catalog::ColumnEntry],
    capacity: usize,
) -> Vec<crate::batch::ColumnBuilder> {
    output_columns
        .iter()
        .map(|col| {
            let encrypted = table_columns
                .iter()
                .any(|c| c.id == col.column_id && c.is_encrypted());
            if encrypted {
                crate::batch::ColumnBuilder::new(TypeId::Bytea, capacity)
            } else {
                let phys = TypeId::timestamp_physical_type_id(col.type_id, col.fractional_digits);
                if phys != col.type_id || col.fractional_digits.is_some() {
                    crate::batch::ColumnBuilder::new_ts(
                        col.type_id,
                        phys,
                        col.fractional_digits,
                        capacity,
                    )
                } else {
                    crate::batch::ColumnBuilder::new(col.type_id, capacity)
                }
            }
        })
        .collect()
}

/// Resolves a branch's append overlay file id and page count for a table, or
/// (None, 0) on the main line. The scan reads this range after the main range.
fn branch_append_range(
    ctx: &ExecutionContext,
    branch_id: Option<u64>,
    heap_file_id: u32,
) -> (Option<u32>, u64) {
    match (branch_id, &ctx.branch_catalog) {
        (Some(bid), Some(cat)) => {
            let files = cat.branch_files_for(bid, heap_file_id);
            (
                Some(files.append_file_id),
                cat.append_page_count(bid, heap_file_id),
            )
        }
        _ => (None, 0),
    }
}

/// Minimum number of pages before parallel scan is used.
/// Below this threshold, the task spawn overhead outweighs the benefit.
const PARALLEL_SCAN_MIN_PAGES: u64 = 64;

// ---------------------------------------------------------------------------
// Sequential scan
// ---------------------------------------------------------------------------

/// Reads all visible tuples from a heap file, one page at a time.
/// Produces DataBatch batches of up to batch_size rows. An optional
/// predicate is evaluated after decoding and applied as a post-filter.
pub struct SeqScanOperator {
    ctx: Arc<ExecutionContext>,
    table_entry: Arc<TableEntry>,
    output_columns: Vec<LogicalColumn>,
    /// Per-table-column index into `output_columns`. Built once at
    /// construction so the per-row decoder does an O(1) lookup instead of
    /// scanning the projection list.
    column_to_builder: Vec<Option<u16>>,
    predicate: Option<BoundExpr>,
    page_cursor: u64,
    /// Resume position within the current page when a previous next() call
    /// stopped mid-page after filling its output batch. Zero means start from
    /// the first slot of the page identified by page_cursor.
    slot_cursor: u16,
    num_pages: u64,
    finished: bool,
    track_tuple_ids: bool,
    /// When set, use version-based visibility instead of MVCC snapshot.
    as_of_version: Option<u64>,
    /// Effective branch for this scan. When set, each page id is resolved
    /// through the branch override chain before reading.
    branch_id: Option<u64>,
    /// Append overlay file id for the active branch, holding rows the branch
    /// inserted. None on the main line. Scanned as a second range after the
    /// main range.
    branch_append_file_id: Option<u32>,
    /// Number of pages in the branch append file to scan.
    num_append_pages: u64,
    /// False while scanning the main range, true once scanning the append range.
    in_append_phase: bool,
    /// When true the main range is skipped entirely and only the branch append
    /// range is scanned. Used by the branch-aware index scan to read the insert
    /// delta after draining the main index.
    append_only: bool,
    /// This table's IO counters, resolved once at construction. Updated once
    /// per batch with the rows produced and the page bytes read to produce them.
    io_stats: Option<Arc<zyron_common::TableIOStats>>,
}

impl SeqScanOperator {
    /// Creates a new sequential scan operator for the given table.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        track_tuple_ids: bool,
        as_of_version: Option<u64>,
    ) -> Result<Self> {
        let branch_id = ctx.active_branch_id;
        let table_entry = ctx.get_table_entry(table_id)?;
        // cached atomic load instead of disk_manager.num_pages which would
        // queue on the per-file tokio Mutex under concurrency
        let hf = ctx.get_heap_file(table_id).await?;
        let num_pages = hf.num_pages_cached() as u64;
        let output_ids: Vec<zyron_catalog::ColumnId> =
            columns.iter().map(|c| c.column_id).collect();
        let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);
        let (branch_append_file_id, num_append_pages) =
            branch_append_range(&ctx, branch_id, table_entry.heap_file_id);
        let io_stats = ctx.table_io_stats_for(table_id.0);
        if let Some(stats) = &io_stats {
            stats.record_seq_scan();
        }

        Ok(Self {
            ctx,
            table_entry,
            output_columns: columns,
            column_to_builder,
            predicate,
            page_cursor: 0,
            slot_cursor: 0,
            num_pages,
            finished: false,
            track_tuple_ids,
            as_of_version,
            branch_id,
            branch_append_file_id,
            num_append_pages,
            in_append_phase: false,
            append_only: false,
            io_stats,
        })
    }

    /// Overrides the scan's branch (used for a per-query `IN BRANCH name` that
    /// differs from the session's active branch). Recomputes the append range
    /// for the new branch.
    pub fn with_branch(mut self, branch_id: Option<u64>) -> Self {
        self.branch_id = branch_id;
        let (af, np) = branch_append_range(&self.ctx, branch_id, self.table_entry.heap_file_id);
        self.branch_append_file_id = af;
        self.num_append_pages = np;
        self
    }

    /// Restricts the scan to the branch append range, skipping the main range.
    /// The branch-aware index scan uses this to read the insert delta the main
    /// index does not cover.
    pub fn append_only(mut self) -> Self {
        self.append_only = true;
        self.in_append_phase = true;
        self
    }

    /// Points the scan at an explicit append file range, skipping the main
    /// range, independent of any session branch. MERGE uses this to read a
    /// branch's inserted rows so it can replay them onto the main line.
    pub fn scan_append_file(mut self, append_file_id: u32, num_pages: u64) -> Self {
        self.branch_append_file_id = Some(append_file_id);
        self.num_append_pages = num_pages;
        self.append_only = true;
        self.in_append_phase = true;
        self
    }

    /// Enforces column-level security on a result batch. Delegates to the
    /// shared operator-level policy so heap and columnar scans behave
    /// identically.
    fn apply_column_security(&self, batch: DataBatch) -> DataBatch {
        crate::operator::apply_column_security(
            &self.ctx,
            self.table_entry.id.0,
            &self.output_columns,
            batch,
        )
    }
}

impl Operator for SeqScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let _scan = zyron_common::profile::scope(zyron_common::profile::Phase::ExecSeqScanNext);
            if self.finished {
                return Ok(None);
            }
            self.ctx.check_cancelled()?;

            let mut timer = crate::calibrate::BatchTimer::start(
                zyron_pressure::capability::OperatorKind::SeqScan,
            );
            let batch_size = self.ctx.batch_size;
            let count_only =
                self.output_columns.is_empty() && self.predicate.is_none() && !self.track_tuple_ids;
            let mut builders =
                scan_builders(&self.output_columns, &self.table_entry.columns, batch_size);
            let mut tuple_ids: Vec<TupleId> = if self.track_tuple_ids {
                Vec::with_capacity(batch_size)
            } else {
                Vec::new()
            };
            let mut row_count: usize = 0;
            // Pages fetched for this batch, accumulated locally and folded into
            // the table counters once when the batch is done. A page revisited
            // because a previous batch filled mid-page counts again, which is
            // what happened: it was fetched again.
            let mut pages_read: u64 = 0;

            while row_count < batch_size {
                // The main range resolves each page through the branch override
                // chain; once it is exhausted the scan continues into the
                // branch append range, read directly with snapshot visibility.
                let page_id = if !self.in_append_phase {
                    if self.page_cursor >= self.num_pages {
                        if self.branch_append_file_id.is_some() && self.num_append_pages > 0 {
                            self.in_append_phase = true;
                            self.page_cursor = 0;
                            self.slot_cursor = 0;
                            continue;
                        }
                        break;
                    }
                    self.ctx.resolve_branch_page(
                        self.branch_id,
                        PageId::new(self.table_entry.heap_file_id, self.page_cursor),
                    )
                } else {
                    if self.page_cursor >= self.num_append_pages {
                        break;
                    }
                    PageId::new(self.branch_append_file_id.unwrap(), self.page_cursor)
                };

                let page_data: [u8; PAGE_SIZE] =
                    read_page_through_pool(&self.ctx.buffer_pool, &self.ctx.disk_manager, page_id)
                        .await?;
                pages_read += 1;

                // Empty page fast path, avoid HeapPage box allocation when
                // the page has zero slots (freshly allocated, never written)
                let header = HeapPage::heap_header_from_slice(&page_data);
                if header.slot_count == 0 {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                    continue;
                }

                // Tuple views borrow straight from the stack copy, no boxed
                // HeapPage and no second 8KB move per page
                let slot_count = header.slot_count;
                let mut slot_idx = self.slot_cursor;
                let mut filled_batch = false;

                while slot_idx < slot_count {
                    let slot_id = zyron_storage::SlotId(slot_idx);
                    slot_idx += 1;
                    let Some(tuple) = HeapPage::get_tuple_view_from_slice(&page_data, slot_id)
                    else {
                        continue;
                    };
                    if tuple.is_deleted() {
                        continue;
                    }
                    // Time-travel visibility dates each tuple by its
                    // transactions' commit LSNs: visible at version N when the
                    // inserter committed at an LSN <= N and the deleter (if any)
                    // committed at an LSN > N. This reconstructs the committed
                    // state as of N from the MVCC versions the heap already
                    // holds. Normal queries use live-snapshot MVCC visibility.
                    let hdr = tuple.header;
                    if let Some(target_version) = self.as_of_version {
                        if !self.ctx.snapshot.status_map().is_visible_at_version(
                            hdr.xmin as u64,
                            hdr.xmax as u64,
                            target_version,
                        ) {
                            continue;
                        }
                    } else if !hdr.is_visible_to(&self.ctx.snapshot) {
                        continue;
                    }

                    if !count_only {
                        decode_tuple_into_builders(
                            tuple.data,
                            &self.table_entry.columns,
                            &self.column_to_builder,
                            &mut builders,
                        );

                        if self.track_tuple_ids {
                            tuple_ids.push(TupleId::new(page_id, slot_id.0));
                        }
                    }

                    row_count += 1;
                    if row_count >= batch_size {
                        filled_batch = true;
                        break;
                    }
                }

                if filled_batch && slot_idx < slot_count {
                    // Resume from slot_idx on this same page in the next batch
                    self.slot_cursor = slot_idx;
                } else {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                }
            }

            if let Some(stats) = &self.io_stats {
                stats.record_seq_batch(row_count as u64, pages_read * PAGE_SIZE as u64);
            }
            // Set once, ahead of every return below. The timer records when it
            // drops, so each exit path reports the rows it actually produced
            // without a call of its own
            timer.rows(row_count as u64);

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            // Count-only path emits the visible-row count with no column data.
            if count_only {
                return Ok(Some(ExecutionBatch::new(DataBatch::with_row_count(
                    row_count,
                ))));
            }

            let mut batch = finalize_builders(builders);

            // ENCRYPTED columns decrypt before anything reads the batch, so
            // predicates evaluate over plaintext. Pushdown into encrypted
            // storage is impossible, this is where the fallback lands
            decrypt_encrypted_columns(
                &self.ctx,
                &mut batch,
                &self.table_entry.columns,
                &self.output_columns,
            )?;
            // Media descriptors inflate back to their original payloads
            inflate_media_columns(
                &self.ctx,
                &mut batch,
                &self.table_entry.columns,
                &self.output_columns,
            )?;
            let batch = batch;

            // Apply predicate filter if present. The predicate runs on the
            // real (unmasked) values; column-level security is applied to the
            // surviving rows afterward so masking never changes filtering.
            if let Some(ref predicate) = self.predicate {
                let mask_col = evaluate(predicate, &batch, &self.output_columns, &self.ctx.params)?;
                let mask = column_to_mask(&mask_col);

                let filtered = batch.filter(&mask);
                let secured = self.apply_column_security(filtered);

                if self.track_tuple_ids {
                    let filtered_ids: Vec<TupleId> = mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| if keep { Some(tuple_ids[i]) } else { None })
                        .collect();
                    return Ok(Some(ExecutionBatch::with_tuple_ids(secured, filtered_ids)));
                }

                return Ok(Some(ExecutionBatch::new(secured)));
            }

            let secured = self.apply_column_security(batch);
            if self.track_tuple_ids {
                Ok(Some(ExecutionBatch::with_tuple_ids(secured, tuple_ids)))
            } else {
                Ok(Some(ExecutionBatch::new(secured)))
            }
        })
    }
}

/// Decrypts the ENCRYPTED columns present in a scan's output batch, turning
/// stored ciphertext back into the column's logical text value. A missing
/// key store or key fails the scan loudly rather than serving ciphertext
pub(crate) fn decrypt_encrypted_columns(
    ctx: &ExecutionContext,
    batch: &mut DataBatch,
    table_columns: &[zyron_catalog::ColumnEntry],
    output_columns: &[LogicalColumn],
) -> Result<()> {
    for (idx, out_col) in output_columns.iter().enumerate() {
        let Some(entry) = table_columns.iter().find(|c| c.id == out_col.column_id) else {
            continue;
        };
        if !entry.is_encrypted() || idx >= batch.columns.len() {
            continue;
        }
        let Some(store) = ctx.key_store.as_ref() else {
            return Err(zyron_common::ZyronError::ExecutionError(format!(
                "column {} is ENCRYPTED but the server has no key store",
                entry.name
            )));
        };
        let key = store.get_key(entry.attrs.encryption_key_id)?;
        let algorithm = match entry.attrs.encryption_algorithm {
            0 => zyron_auth::EncryptionAlgorithm::Aes128Gcm,
            1 => zyron_auth::EncryptionAlgorithm::Aes256Gcm,
            other => {
                return Err(zyron_common::ZyronError::ExecutionError(format!(
                    "column {} declares unknown encryption algorithm {other}",
                    entry.name
                )));
            }
        };
        let mut aad = [0u8; 6];
        aad[..4].copy_from_slice(&entry.table_id.0.to_le_bytes());
        aad[4..].copy_from_slice(&entry.id.0.to_le_bytes());
        let source = &batch.columns[idx];
        let rows = batch.num_rows;
        let mut out = Vec::with_capacity(rows);
        let mut nulls = crate::column::NullBitmap::none(rows);
        for row in 0..rows {
            if source.nulls.is_null(row) {
                nulls.set_null(row);
                out.push(String::new());
                continue;
            }
            let ciphertext = match source.get_scalar(row) {
                crate::column::ScalarValue::Binary(b) => b,
                other => {
                    return Err(zyron_common::ZyronError::ExecutionError(format!(
                        "encrypted column {} holds unexpected stored value {other:?}",
                        entry.name
                    )));
                }
            };
            let plaintext =
                zyron_auth::encryption::decrypt_value(&ciphertext, &key, algorithm, &aad)?;
            out.push(String::from_utf8(plaintext).map_err(|_| {
                zyron_common::ZyronError::ExecutionError(format!(
                    "decrypted value of column {} is not valid text",
                    entry.name
                ))
            })?);
        }
        batch.columns[idx] = crate::column::Column::with_nulls(
            crate::column::ColumnData::Utf8(out),
            nulls,
            out_col.type_id,
        );
    }
    Ok(())
}

/// Inflates media descriptors in a scan's output back to the original
/// payload bytes: inline descriptors carry them, TOAST and external ones
/// read the content addressed store, and URI references fetch through the
/// external fetcher. A SELECT of a media column always answers with the
/// bytes that were inserted
pub(crate) fn inflate_media_columns(
    ctx: &ExecutionContext,
    batch: &mut DataBatch,
    table_columns: &[zyron_catalog::ColumnEntry],
    output_columns: &[LogicalColumn],
) -> Result<()> {
    use zyron_common::TypeId as T;
    for (idx, out_col) in output_columns.iter().enumerate() {
        if !matches!(
            out_col.type_id,
            T::Image | T::Video | T::Audio | T::Document | T::ExternalRef
        ) || idx >= batch.columns.len()
        {
            continue;
        }
        let Some(entry) = table_columns.iter().find(|c| c.id == out_col.column_id) else {
            continue;
        };
        let source = &batch.columns[idx];
        let rows = batch.num_rows;
        let mut out = Vec::with_capacity(rows);
        let mut nulls = crate::column::NullBitmap::none(rows);
        for row in 0..rows {
            if source.nulls.is_null(row) {
                nulls.set_null(row);
                out.push(Vec::new());
                continue;
            }
            let stored = match source.get_scalar(row) {
                ScalarValue::Binary(b) => b,
                other => {
                    return Err(zyron_common::ZyronError::ExecutionError(format!(
                        "media column {} holds unexpected stored value {other:?}",
                        entry.name
                    )));
                }
            };
            if !zyron_media::descriptor::is_descriptor(&stored) {
                // A pre-descriptor payload reads back as it was stored
                out.push(stored);
                continue;
            }
            let (descriptor, inline_payload) =
                zyron_media::descriptor::MediaDescriptor::from_bytes(&stored)
                    .map_err(zyron_common::ZyronError::from)?;
            use zyron_media::descriptor::StorageMode;
            let payload = match descriptor.mode {
                StorageMode::Inline => inline_payload.ok_or_else(|| {
                    zyron_common::ZyronError::ExecutionError(format!(
                        "inline media descriptor of column {} carries no payload",
                        entry.name
                    ))
                })?,
                StorageMode::Toast | StorageMode::External => {
                    let Some(store) = ctx.media_store.as_ref() else {
                        return Err(zyron_common::ZyronError::ExecutionError(format!(
                            "column {} needs the media store, which only a running server opens",
                            entry.name
                        )));
                    };
                    store
                        .get(&descriptor.sha256)
                        .map_err(zyron_common::ZyronError::from)?
                }
                StorageMode::ExternalUri => {
                    let uri = descriptor.uri.as_deref().ok_or_else(|| {
                        zyron_common::ZyronError::ExecutionError(format!(
                            "external reference of column {} carries no URI",
                            entry.name
                        ))
                    })?;
                    crate::media_runtime::external_fetcher()
                        .fetch(uri)
                        .map_err(zyron_common::ZyronError::from)?
                }
            };
            out.push(payload);
        }
        batch.columns[idx] = crate::column::Column::with_nulls(
            crate::column::ColumnData::Binary(out),
            nulls,
            out_col.type_id,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parallel sequential scan
// ---------------------------------------------------------------------------

/// Multi-threaded sequential scan that divides the page range across
/// multiple tokio tasks. Each worker scans its assigned pages, decodes
/// visible tuples, applies the predicate, and sends result batches
/// through an MPSC channel. The operator's next() receives from the
/// channel, providing multi-core throughput for large table scans.
///
/// Not used for tuple ID tracking (DML operations need ordered IDs).
pub struct ParallelSeqScanOperator {
    receiver: tokio::sync::mpsc::Receiver<Result<DataBatch>>,
    finished: bool,
    /// Worker join handles, retained so a worker that panics (rather than
    /// returning Err through the channel) is detected as an error instead of
    /// being mistaken for clean end-of-stream, which would silently truncate
    /// the result set.
    workers: Vec<crate::parallel_pool::JoinHandle<()>>,
    /// Parallel work permits held for as long as the workers run. Dropping
    /// this hands the machine's capacity back to whatever query asks next
    _grant: crate::parallel_pool::DopGrant,
}

impl ParallelSeqScanOperator {
    /// Creates a parallel scan operator. Spawns worker tasks immediately.
    /// Each worker scans a contiguous slice of the table's pages.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
    ) -> Result<Self> {
        let table_entry = ctx.get_table_entry(table_id)?;
        let num_pages = ctx.get_heap_file(table_id).await?.num_pages_cached() as u64;
        // One scan, however many workers divide it. Each worker's scanner folds
        // in its own row and byte totals.
        if let Some(stats) = ctx.table_io_stats_for(table_id.0) {
            stats.record_seq_scan();
        }

        // The page count bounds the split the data supports, at a floor of
        // work per worker below which the wake costs more than the work.
        // What the machine can currently afford is a different question,
        // and the pool answers it: a scan planned while fifty others are
        // running is handed fewer workers than the same scan on an idle node
        let natural_workers = parallel_workers_for_pages(num_pages);
        let grant = crate::parallel_pool::reserve(natural_workers);
        let num_workers = grant.workers().min(natural_workers).max(1);

        // Channel capacity: 2 batches per worker to keep workers busy
        // without unbounded buffering.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<DataBatch>>(num_workers * 2);

        // Every worker claims runs of pages from one cursor rather than
        // owning a fixed slice, so the scan ends when the pages run out
        // and not when the slowest worker finishes its share
        let claims = Arc::new(PageClaims::new(0, num_pages));
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let tx = tx.clone();
            let ctx = ctx.clone();
            let table_entry = table_entry.clone();
            let columns = columns.clone();
            let predicate = predicate.clone();
            let claims = Arc::clone(&claims);

            // The shared pool, not the current runtime. On the serving path
            // the current runtime drives one connection, so spawning there
            // would put every worker on the one thread and the split would
            // buy nothing
            workers.push(crate::parallel_pool::spawn(async move {
                let result = scan_page_range(
                    &ctx,
                    &table_entry,
                    &columns,
                    predicate.as_ref(),
                    claims,
                    &tx,
                )
                .await;

                // If the scan itself errored, send the error through the channel.
                if let Err(e) = result {
                    let _ = tx.send(Err(e)).await;
                }
            }));
        }

        Ok(Self {
            receiver: rx,
            finished: false,
            workers,
            _grant: grant,
        })
    }
}

/// Pages of a heap file below which one more worker is not worth waking.
///
/// Waking a parked pool thread costs about ten microseconds on the
/// spawner's side and the thread starts some tens of microseconds later,
/// against a few microseconds of decode per page. Thirty two pages is a
/// few hundred microseconds of work, enough that the wake is a small part
/// of it
pub(crate) const PARALLEL_SCAN_MIN_PAGES_PER_WORKER: u64 = 32;

/// Pages a worker takes from the shared cursor at a time.
///
/// Small enough that the tail of the scan is spread evenly over the
/// workers still running, large enough that the claim is a negligible
/// share of the work it hands over
pub(crate) const PARALLEL_SCAN_CLAIM_PAGES: u64 = 8;

/// Workers a heap scan of `num_pages` pages naturally splits into
pub(crate) fn parallel_workers_for_pages(num_pages: u64) -> usize {
    (num_pages
        .div_ceil(PARALLEL_SCAN_MIN_PAGES_PER_WORKER)
        .max(1)) as usize
}

/// Pages the workers of one fan-out claim as they go.
///
/// A worker that owns a fixed slice of the table sets the scan's wall
/// time when it is the slow one, and on a machine with two kinds of core
/// one of them always is. Claiming runs from a shared cursor means a fast
/// worker takes more of the table, a slow one less, and a worker whose
/// thread started late finds less left rather than holding a share the
/// others cannot touch
pub(crate) struct PageClaims {
    next: std::sync::atomic::AtomicU64,
    end: u64,
}

impl PageClaims {
    pub(crate) fn new(start: u64, end: u64) -> Self {
        Self {
            next: std::sync::atomic::AtomicU64::new(start),
            end,
        }
    }

    /// The next run of up to `chunk` pages, None once every page is taken
    fn claim(&self, chunk: u64) -> Option<(u64, u64)> {
        let start = self
            .next
            .fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
        if start >= self.end {
            return None;
        }
        Some((start, start.saturating_add(chunk).min(self.end)))
    }
}

/// Scans pages claimed from the shared cursor, decodes visible tuples,
/// applies the predicate filter, and sends result batches through the
/// channel.
async fn scan_page_range(
    ctx: &ExecutionContext,
    table_entry: &TableEntry,
    output_columns: &[LogicalColumn],
    predicate: Option<&BoundExpr>,
    claims: Arc<PageClaims>,
    tx: &tokio::sync::mpsc::Sender<Result<DataBatch>>,
) -> Result<()> {
    let mut scanner =
        PageRangeScanner::claiming(ctx, table_entry, output_columns, predicate, claims);
    while let Some(batch) = scanner.next_batch().await? {
        if tx.send(Ok(batch)).await.is_err() {
            break;
        }
    }
    Ok(())
}

/// Pull-based scanner over a page range, fixed or claimed run by run from
/// a shared cursor. The single decode/visibility/count-only path shared by
/// the parallel scan and the parallel aggregate, so both consume rows
/// identically.
///
/// Count-only mirrors SeqScanOperator. When no columns are projected and no
/// predicate filters rows, COUNT(*) needs only the visible-row count, so the
/// batch carries num_rows with no column data. Without this the empty builders
/// finalize to a zero-column batch whose num_rows is 0, dropping every row.
pub(crate) struct PageRangeScanner<'a> {
    ctx: &'a ExecutionContext,
    table_entry: &'a TableEntry,
    output_columns: &'a [LogicalColumn],
    predicate: Option<&'a BoundExpr>,
    column_to_builder: Vec<Option<u16>>,
    count_only: bool,
    page_cursor: u64,
    end_page: u64,
    /// Where the next run of pages comes from once the current one is
    /// spent. None for a scanner given one fixed range
    claims: Option<Arc<PageClaims>>,
    // Resume position within the current page when a batch fills mid-page.
    // Without this, slots after the break would be skipped because page_cursor
    // already advanced.
    slot_cursor: u16,
    /// This table's IO counters. Each worker holds its own Arc to the same
    /// entry and folds its batch totals in, so the table's counters are the sum
    /// across workers. Scan initiation is recorded by the owning operator, once
    /// for the whole parallel scan rather than once per worker.
    io_stats: Option<Arc<zyron_common::TableIOStats>>,
}

impl<'a> PageRangeScanner<'a> {
    pub(crate) fn new(
        ctx: &'a ExecutionContext,
        table_entry: &'a TableEntry,
        output_columns: &'a [LogicalColumn],
        predicate: Option<&'a BoundExpr>,
        start_page: u64,
        end_page: u64,
    ) -> Self {
        let output_ids: Vec<zyron_catalog::ColumnId> =
            output_columns.iter().map(|c| c.column_id).collect();
        let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);
        let count_only = output_columns.is_empty() && predicate.is_none();
        let io_stats = ctx.table_io_stats_for(table_entry.id.0);
        Self {
            ctx,
            table_entry,
            output_columns,
            predicate,
            column_to_builder,
            count_only,
            page_cursor: start_page,
            end_page,
            claims: None,
            slot_cursor: 0,
            io_stats,
        }
    }

    /// A scanner over runs of pages claimed from a cursor shared with the
    /// other workers of one fan-out
    pub(crate) fn claiming(
        ctx: &'a ExecutionContext,
        table_entry: &'a TableEntry,
        output_columns: &'a [LogicalColumn],
        predicate: Option<&'a BoundExpr>,
        claims: Arc<PageClaims>,
    ) -> Self {
        let mut scanner = Self::new(ctx, table_entry, output_columns, predicate, 0, 0);
        scanner.claims = Some(claims);
        scanner
    }

    /// Whether a page remains to read, claiming the next run from the
    /// shared cursor when the current one is spent
    fn page_pending(&mut self) -> bool {
        if self.page_cursor < self.end_page {
            return true;
        }
        let Some(claims) = &self.claims else {
            return false;
        };
        match claims.claim(PARALLEL_SCAN_CLAIM_PAGES) {
            Some((start, end)) => {
                self.page_cursor = start;
                self.end_page = end;
                self.slot_cursor = 0;
                true
            }
            None => false,
        }
    }

    /// Produces the next result batch, or None when the range is exhausted.
    pub(crate) async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        let batch_size = self.ctx.batch_size;

        while self.page_pending() {
            self.ctx.check_cancelled()?;

            let mut builders =
                scan_builders(self.output_columns, &self.table_entry.columns, batch_size);
            let mut row_count = 0usize;
            // Pages fetched for this batch, folded into the table counters once
            // when the batch is done rather than once per page.
            let mut pages_read: u64 = 0;

            while row_count < batch_size && self.page_pending() {
                let page_id = self.ctx.resolve_branch_page(
                    self.ctx.active_branch_id,
                    PageId::new(self.table_entry.heap_file_id, self.page_cursor),
                );

                let page_data: [u8; PAGE_SIZE] =
                    read_page_through_pool(&self.ctx.buffer_pool, &self.ctx.disk_manager, page_id)
                        .await?;
                pages_read += 1;

                let header = HeapPage::heap_header_from_slice(&page_data);
                if header.slot_count == 0 {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                    continue;
                }

                // Tuple views borrow straight from the stack copy, no boxed
                // HeapPage and no second 8KB move per page
                let slot_count = header.slot_count;
                let mut slot_idx = self.slot_cursor;
                let mut filled_batch = false;

                while slot_idx < slot_count {
                    let slot_id = zyron_storage::SlotId(slot_idx);
                    slot_idx += 1;
                    let Some(tuple) = HeapPage::get_tuple_view_from_slice(&page_data, slot_id)
                    else {
                        continue;
                    };
                    if tuple.is_deleted() {
                        continue;
                    }
                    let hdr = tuple.header;
                    if !hdr.is_visible_to(&self.ctx.snapshot) {
                        continue;
                    }

                    if !self.count_only {
                        decode_tuple_into_builders(
                            tuple.data,
                            &self.table_entry.columns,
                            &self.column_to_builder,
                            &mut builders,
                        );
                    }

                    row_count += 1;
                    if row_count >= batch_size {
                        filled_batch = true;
                        break;
                    }
                }

                if filled_batch && slot_idx < slot_count {
                    self.slot_cursor = slot_idx;
                } else {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                }
            }

            if let Some(stats) = &self.io_stats {
                stats.record_seq_batch(row_count as u64, pages_read * PAGE_SIZE as u64);
            }

            if row_count == 0 {
                break;
            }

            if self.count_only {
                return Ok(Some(DataBatch::with_row_count(row_count)));
            }

            let mut batch = finalize_builders(builders);
            // ENCRYPTED columns decrypt before anything reads the batch, so
            // the predicate below evaluates over plaintext
            decrypt_encrypted_columns(
                self.ctx,
                &mut batch,
                &self.table_entry.columns,
                self.output_columns,
            )?;
            let batch = batch;
            if let Some(pred) = self.predicate {
                let mask_col = evaluate(pred, &batch, self.output_columns, &self.ctx.params)?;
                let mask = column_to_mask(&mask_col);
                let filtered = batch.filter(&mask);
                if filtered.num_rows == 0 {
                    continue;
                }
                return Ok(Some(filtered));
            }
            return Ok(Some(batch));
        }

        Ok(None)
    }
}

impl Operator for ParallelSeqScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let _scan =
                zyron_common::profile::scope(zyron_common::profile::Phase::ExecParallelScanNext);
            if self.finished {
                return Ok(None);
            }

            match self.receiver.recv().await {
                Some(Ok(batch)) => Ok(Some(ExecutionBatch::new(batch))),
                Some(Err(e)) => {
                    self.finished = true;
                    Err(e)
                }
                None => {
                    self.finished = true;
                    // The channel closed because every worker's sender dropped.
                    // Distinguish clean completion from a worker that panicked
                    // (which would otherwise look like an early end-of-stream
                    // and silently truncate results). Join the handles and
                    // surface a panic as an execution error.
                    for handle in self.workers.drain(..) {
                        if handle.await.is_err() {
                            return Err(zyron_common::ZyronError::ExecutionError(
                                "parallel scan worker panicked".to_string(),
                            ));
                        }
                    }
                    Ok(None)
                }
            }
        })
    }
}

/// Determines whether a parallel scan should be used for the given table.
/// Returns true when the table has enough pages to benefit from parallelism
/// and tuple ID tracking is not required.
pub fn should_use_parallel_scan(num_pages: u64, track_tuple_ids: bool) -> bool {
    !track_tuple_ids && num_pages >= PARALLEL_SCAN_MIN_PAGES
}

// ---------------------------------------------------------------------------
// Index scan
// ---------------------------------------------------------------------------

/// Key range bounds extracted from an index predicate for B+ tree lookup.
struct ScanBounds {
    start_key: Option<Vec<u8>>,
    end_key: Option<Vec<u8>>,
}

/// Serializes a LiteralValue to big-endian bytes for B+ tree key comparison.
/// Returns None for types that cannot be used as index keys.
fn literal_to_key_bytes(value: &LiteralValue) -> Option<Vec<u8>> {
    match value {
        // Sign bit flipped to match the indexer, so negatives order below
        // positives under the tree's unsigned bytewise comparison
        LiteralValue::Integer(v) => Some(((*v as u64) ^ (1u64 << 63)).to_be_bytes().to_vec()),
        // Matches the sixteen byte key the indexer writes for a 128 bit
        // value, same sign bit flip at the wider width
        LiteralValue::Int128(v) => Some(((*v as u128) ^ (1u128 << 127)).to_be_bytes().to_vec()),
        // A decimal index key is scaled to the column's scale, which this
        // literal's own scale need not match, so no key is offered and the
        // predicate is answered by the scan filter instead
        LiteralValue::Decimal { .. } => None,
        LiteralValue::Float(v) => {
            // IEEE 754 float-to-sortable-bytes encoding.
            let bits = v.to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            Some(sortable.to_be_bytes().to_vec())
        }
        LiteralValue::String(s) => Some(s.as_bytes().to_vec()),
        LiteralValue::Boolean(b) => Some(vec![*b as u8]),
        LiteralValue::Null => None,
        // Interval has no order-preserving fixed encoding, so no bound is
        // built and CREATE INDEX refuses interval key columns
        LiteralValue::Interval(_) => None,
        // A range reaches an index in its order preserving form, which is a
        // reordering of the stored bytes this literal carries, so no bound
        // is offered from here and the predicate is answered by the scan
        // filter instead. Seeking with the stored form would look in the
        // wrong place and miss rows
        LiteralValue::Bytes(_) => None,
    }
}

/// Extracts start/end key bounds from an index predicate.
/// Handles equality, less-than, greater-than, and BETWEEN on a single
/// column matching the first column of the index.
///
/// Predicates that cannot be decomposed into range bounds (complex AND
/// trees, OR, functions) return an unbounded scan, letting the remaining
/// predicate handle correctness via post-filtering.
fn extract_scan_bounds(
    predicate: &BoundExpr,
    index: &IndexEntry,
    params: &[ScalarValue],
) -> ScanBounds {
    if index.columns.is_empty() {
        return ScanBounds {
            start_key: None,
            end_key: None,
        };
    }
    let index_col_id = index.columns[0].column_id;

    match predicate {
        // col = literal or literal = col
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_literal(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: Some(bytes.clone()),
                    end_key: Some(bytes),
                };
            }
        }
        // col > literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Gt,
            right,
            ..
        } => {
            // An exclusive bound scans from the boundary value itself. Byte
            // arithmetic on the key is wrong for variable-length values
            // (incrementing 'abc' to 'abd' skips 'abcd'), and the predicate
            // is always re-applied as a post-filter, so the boundary value's
            // own rows drop there
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: Some(bytes),
                    end_key: None,
                };
            }
            // literal > col means col < literal
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: None,
                    end_key: Some(bytes),
                };
            }
        }
        // col >= literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::GtEq,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: Some(bytes),
                    end_key: None,
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: None,
                    end_key: Some(bytes),
                };
            }
        }
        // col < literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Lt,
            right,
            ..
        } => {
            // Same boundary-value rule as Gt: the bound stays at the value
            // and the post-filter excludes the value's own rows
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: None,
                    end_key: Some(bytes),
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: Some(bytes),
                    end_key: None,
                };
            }
        }
        // col <= literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::LtEq,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: None,
                    end_key: Some(bytes),
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: Some(bytes),
                    end_key: None,
                };
            }
        }
        // col BETWEEN low AND high
        BoundExpr::Between {
            expr,
            low,
            high,
            negated: false,
        } => {
            if matches_index_column(expr, index_col_id) {
                let col_ty = column_type_id(expr);
                let start = extract_constant_bytes(low, params, col_ty);
                let end = extract_constant_bytes(high, params, col_ty);
                if start.is_some() || end.is_some() {
                    return ScanBounds {
                        start_key: start,
                        end_key: end,
                    };
                }
            }
        }
        // AND: intersect bounds from both sides
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            let left_bounds = extract_scan_bounds(left, index, params);
            let right_bounds = extract_scan_bounds(right, index, params);
            return ScanBounds {
                start_key: pick_later_key(left_bounds.start_key, right_bounds.start_key),
                end_key: pick_earlier_key(left_bounds.end_key, right_bounds.end_key),
            };
        }
        _ => {}
    }

    ScanBounds {
        start_key: None,
        end_key: None,
    }
}

/// Returns true if the expression is a ColumnRef matching the given column ID.
fn matches_index_column(expr: &BoundExpr, col_id: zyron_catalog::ColumnId) -> bool {
    matches!(expr, BoundExpr::ColumnRef(cr) if cr.column_id == col_id)
}

/// Returns the TypeId carried by a ColumnRef expression when one is
/// present, otherwise None. Used so index bound encoding can coerce a
/// parameter scalar to the same byte layout the indexer used at INSERT
/// time even when the wire layer decoded the parameter as Utf8 (which
/// happens whenever the client sent Parse with zero param type hints).
fn column_type_id(expr: &BoundExpr) -> Option<TypeId> {
    if let BoundExpr::ColumnRef(cr) = expr {
        Some(cr.type_id)
    } else {
        None
    }
}

/// Checks if left is a ColumnRef matching col_id and right is a constant.
/// Returns the constant serialized as key bytes.
fn match_column_op_literal(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
    params: &[ScalarValue],
) -> Option<Vec<u8>> {
    if matches_index_column(left, col_id) {
        return extract_constant_bytes(right, params, column_type_id(left));
    }
    None
}

/// Checks if left is a constant and right is a ColumnRef matching col_id.
/// Returns the constant serialized as key bytes.
fn match_literal_op_column(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
    params: &[ScalarValue],
) -> Option<Vec<u8>> {
    if matches_index_column(right, col_id) {
        return extract_constant_bytes(left, params, column_type_id(right));
    }
    None
}

/// Matches col = constant or constant = col patterns.
fn match_column_literal(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
    params: &[ScalarValue],
) -> Option<Vec<u8>> {
    match_column_op_literal(left, right, col_id, params)
        .or_else(|| match_literal_op_column(left, right, col_id, params))
}

/// Extracts constant bytes from a `BoundExpr::Literal` or a
/// `BoundExpr::Parameter` resolved against the executor's bind parameters.
/// `column_ty` is the TypeId of the indexed column being compared, used to
/// coerce a Utf8-decoded parameter back to the column's wire encoding.
fn extract_constant_bytes(
    expr: &BoundExpr,
    params: &[ScalarValue],
    column_ty: Option<TypeId>,
) -> Option<Vec<u8>> {
    match expr {
        BoundExpr::Literal { value, .. } => literal_to_key_bytes(value),
        // PG parameter indexes are 1-based ($1, $2, ...) while the params
        // slice is 0-based. The mismatch silently produced empty bounds and
        // an open range scan for every prepared point lookup.
        BoundExpr::Parameter { index, .. } if *index >= 1 => params
            .get(*index - 1)
            .and_then(|v| scalar_to_key_bytes(v, column_ty)),
        _ => None,
    }
}

/// Encodes a bound parameter into the same big-endian, order-preserving
/// byte layout that `encode_btree_key_into` uses when an indexed row is
/// indexed. `column_ty` is the indexed column's TypeId, used to coerce
/// from the parameter's wire-decoded scalar shape (often Utf8 when the
/// client did not send param type hints during Parse) to the layout the
/// indexer wrote.
fn scalar_to_key_bytes(value: &ScalarValue, column_ty: Option<TypeId>) -> Option<Vec<u8>> {
    if let (ScalarValue::Utf8(s), Some(ty)) = (value, column_ty) {
        if let Some(bytes) = coerce_text_to_key_bytes(s, ty) {
            return Some(bytes);
        }
    }
    match value {
        ScalarValue::Null => None,
        ScalarValue::Boolean(b) => Some(vec![*b as u8]),
        // Sign bit flipped to match the indexer's order-preserving layout
        ScalarValue::Int8(v) => Some(((*v as i64 as u64) ^ (1u64 << 63)).to_be_bytes().to_vec()),
        ScalarValue::Int16(v) => Some(((*v as i64 as u64) ^ (1u64 << 63)).to_be_bytes().to_vec()),
        ScalarValue::Int32(v) => Some(((*v as i64 as u64) ^ (1u64 << 63)).to_be_bytes().to_vec()),
        ScalarValue::Int64(v) => Some(((*v as u64) ^ (1u64 << 63)).to_be_bytes().to_vec()),
        ScalarValue::Int128(v) => {
            let key = (*v as u128) ^ (1u128 << 127);
            Some(key.to_be_bytes().to_vec())
        }
        ScalarValue::UInt8(v) => Some((*v as u64).to_be_bytes().to_vec()),
        ScalarValue::UInt16(v) => Some((*v as u64).to_be_bytes().to_vec()),
        ScalarValue::UInt32(v) => Some((*v as u64).to_be_bytes().to_vec()),
        ScalarValue::UInt64(v) => Some(v.to_be_bytes().to_vec()),
        ScalarValue::Float32(v) => {
            let bits = (*v as f64).to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            Some(sortable.to_be_bytes().to_vec())
        }
        ScalarValue::Float64(v) => {
            let bits = v.to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            Some(sortable.to_be_bytes().to_vec())
        }
        ScalarValue::Utf8(s) => Some(s.as_bytes().to_vec()),
        ScalarValue::Binary(b) => Some(b.clone()),
        ScalarValue::FixedBinary16(b) => Some(b.to_vec()),
        // Interval has no order-preserving fixed encoding, so no bound is
        // built and CREATE INDEX refuses interval key columns
        ScalarValue::Interval(_) => None,
    }
}

/// Coerces a text-encoded parameter to the byte layout the indexer wrote
/// for a column of the given TypeId. Returns None when the text cannot be
/// parsed to that type, which causes the caller to fall back to the raw
/// Utf8 byte path and the index scan to come up empty.
fn coerce_text_to_key_bytes(text: &str, ty: TypeId) -> Option<Vec<u8>> {
    match ty {
        TypeId::Int8 | TypeId::Int16 | TypeId::Int32 | TypeId::Int64 => text
            .trim()
            .parse::<i64>()
            .ok()
            .map(|v| ((v as u64) ^ (1u64 << 63)).to_be_bytes().to_vec()),
        TypeId::UInt8 | TypeId::UInt16 | TypeId::UInt32 | TypeId::UInt64 => text
            .trim()
            .parse::<u64>()
            .ok()
            .map(|v| v.to_be_bytes().to_vec()),
        TypeId::Float32 | TypeId::Float64 => text.trim().parse::<f64>().ok().map(|v| {
            let bits = v.to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            sortable.to_be_bytes().to_vec()
        }),
        TypeId::Boolean => match text.trim() {
            "t" | "T" | "true" | "TRUE" | "1" | "yes" | "on" => Some(vec![1]),
            "f" | "F" | "false" | "FALSE" | "0" | "no" | "off" => Some(vec![0]),
            _ => None,
        },
        _ => None,
    }
}

/// Picks the later (larger) of two optional start keys.
fn pick_later_key(a: Option<Vec<u8>>, b: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a >= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Picks the earlier (smaller) of two optional end keys.
fn pick_earlier_key(a: Option<Vec<u8>>, b: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a <= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Index-guided scan operator. Uses a B+ tree index to look up matching
/// TupleIds, then fetches only those tuples from the heap file.
///
/// When no BTreeIndex instance is registered in the ExecutionContext,
/// falls back to a sequential scan with predicate filtering.
pub struct IndexScanOperator {
    /// B+ tree index scan state. None when falling back to seq scan.
    index_state: Option<IndexScanState>,
    /// Fallback sequential scan when no index instance is available.
    fallback: Option<SeqScanOperator>,
    /// Branch insert delta scan, drained after the main index path. Present only
    /// when a branch is active: the main B+ tree does not index branch-inserted
    /// rows, so they are read from the append range with the predicate applied.
    append_delta: Option<SeqScanOperator>,
}

/// State for an active B+ tree index scan.
struct IndexScanState {
    ctx: Arc<ExecutionContext>,
    table_entry: Arc<TableEntry>,
    output_columns: Vec<LogicalColumn>,
    /// Per-table-column index into `output_columns`, precomputed once.
    column_to_builder: Vec<Option<u16>>,
    remaining_predicate: Option<BoundExpr>,
    track_tuple_ids: bool,
    /// Pre-collected row locators from the B+ tree range scan, heap entries
    /// re-stamped with the table's heap file id.
    locators: Vec<zyron_common::RowLocator>,
    /// Pre-fetched values for columnar-resident entries, batched per segment.
    /// None when every entry is heap resident.
    columnar: Option<crate::operator::doc_fetch::DocRowFetcher>,
    /// Current position in the locators vector.
    cursor: usize,
    /// Active branch for this scan. Main index tids are resolved through this
    /// branch's override chain so rows the branch deleted or modified (their
    /// cow page slot is tombstoned) drop out on fetch.
    branch_id: Option<u64>,
    finished: bool,
    /// Table and index IO counters, updated per batch with the rows fetched
    /// and the page bytes read to fetch them. The scan count and the entries
    /// the range scan examined are recorded when it is built, because the
    /// range scan runs to completion there.
    io_stats: crate::operator::IndexScanStats,
}

impl IndexScanOperator {
    /// Builds a scan over locators the caller already has.
    ///
    /// Replication apply resolves the rows a changeset names itself, by
    /// probing an index with a shipped key or by matching whole row images,
    /// and then needs exactly what an index scan does next: resolve each
    /// locator to a row, apply visibility, batch columnar entries in one pass,
    /// and hand the rows on with their locators attached so a delete can
    /// address them. Rebuilding that would be a second copy of the trickiest
    /// loop in the scan layer, so it is entered here instead
    pub async fn from_locators(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        locators: Vec<zyron_common::RowLocator>,
    ) -> Result<Self> {
        let table_entry = ctx.get_table_entry(table_id)?;
        let has_columnar = locators
            .iter()
            .any(|l| !matches!(l, zyron_common::RowLocator::Heap { .. }));
        let columnar = if has_columnar {
            Some(
                crate::operator::doc_fetch::DocRowFetcher::prepare_columnar_only(
                    &ctx, table_id, &columns, &locators, None,
                )
                .await?,
            )
        } else {
            None
        };
        let output_ids: Vec<zyron_catalog::ColumnId> =
            columns.iter().map(|c| c.column_id).collect();
        let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);
        let io_stats =
            crate::operator::IndexScanStats::open(&ctx, table_id.0, u32::MAX, locators.len());
        let branch_id = ctx.active_branch_id;
        Ok(Self {
            index_state: Some(IndexScanState {
                ctx,
                table_entry,
                output_columns: columns,
                column_to_builder,
                // The caller already decided which rows these are, and a
                // predicate here would be re-deciding it
                remaining_predicate: None,
                track_tuple_ids: true,
                locators,
                columnar,
                cursor: 0,
                branch_id,
                finished: false,
                io_stats,
            }),
            fallback: None,
            append_delta: None,
        })
    }

    /// Creates an index scan operator. When a BTreeIndex instance is
    /// registered in the ExecutionContext for the given index_id, performs
    /// an actual B+ tree range scan. Otherwise falls back to sequential
    /// scan with the predicate applied as a post-filter.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        index: Option<Arc<IndexEntry>>,
        btree: Option<Arc<BTreeIndex>>,
        columns: Vec<LogicalColumn>,
        predicate: BoundExpr,
        remaining_predicate: Option<BoundExpr>,
        track_tuple_ids: bool,
        descending: bool,
    ) -> Result<Self> {
        // Try B+ tree path when both index metadata and a live tree are available.
        if let (Some(index_entry), Some(btree_index)) = (&index, &btree) {
            let bounds = extract_scan_bounds(&predicate, index_entry, &ctx.params);

            // Always re-apply the index predicate as a post-filter, never trust
            // the index entry alone. Under MVCC, deleted rows keep their index
            // entries (filtered here by the per-row visibility check) and a
            // vacuumed-then-reused heap slot can be reached through a stale
            // entry whose key no longer matches the row now in that slot; the
            // post-filter rechecks the actual column value so such a row is
            // dropped. This also covers predicate shapes the bounds extractor
            // cannot decompose (functions, OR, unsupported scalars).
            let effective_remaining = match remaining_predicate {
                Some(rest) => Some(BoundExpr::BinaryOp {
                    left: Box::new(predicate.clone()),
                    op: BinaryOperator::And,
                    right: Box::new(rest),
                    type_id: zyron_common::TypeId::Boolean,
                }),
                None => Some(predicate.clone()),
            };

            let table_entry = ctx.get_table_entry(table_id)?;
            let heap_file_id = table_entry.heap_file_id;

            // A stored key is the leading indexed value, then any further key
            // components, then a locator suffix so duplicate values coexist.
            // Bounds are built on the leading value alone, so the lower bound
            // is the value itself (every entry for it sorts at or above) and
            // the upper bound is the value's successor, which sorts above
            // every entry for it whatever follows in the key.
            let start_key = bounds.start_key.clone();
            let end_key = bounds
                .end_key
                .as_ref()
                .and_then(|k| crate::operator::modify::index_key_upper_bound(k));

            let mut locators: Vec<zyron_common::RowLocator> = Vec::new();
            let mut has_columnar = false;
            btree_index.range_scan_for_each(
                start_key.as_deref(),
                end_key.as_deref(),
                |_key, loc| {
                    match loc {
                        // heap entries store no file id, re-stamp the table's
                        zyron_common::RowLocator::Heap { page, slot } => {
                            locators.push(zyron_common::RowLocator::Heap {
                                page: PageId::new(heap_file_id, page.page_num),
                                slot,
                            });
                        }
                        other => {
                            has_columnar = true;
                            locators.push(other);
                        }
                    }
                    true
                },
            );

            // Columnar-resident entries are pre-fetched in one batched pass
            // through the columnar scan machinery, visibility included
            let columnar = if has_columnar {
                Some(
                    crate::operator::doc_fetch::DocRowFetcher::prepare_columnar_only(
                        &ctx, table_id, &columns, &locators, None,
                    )
                    .await?,
                )
            } else {
                None
            };

            let output_ids: Vec<zyron_catalog::ColumnId> =
                columns.iter().map(|c| c.column_id).collect();
            let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);

            // With a branch active, the main index does not cover rows the
            // branch inserted, so scan the append delta with the full predicate
            // after the main index path. Index scans are only branch-accelerated
            // for reads; DML (track_tuple_ids) stays on the sequential path.
            let branch_id = ctx.active_branch_id;
            let append_delta = if branch_id.is_some() && !track_tuple_ids {
                let full_predicate = match &effective_remaining {
                    Some(rest) => BoundExpr::BinaryOp {
                        left: Box::new(predicate.clone()),
                        op: BinaryOperator::And,
                        right: Box::new(rest.clone()),
                        type_id: zyron_common::TypeId::Boolean,
                    },
                    None => predicate.clone(),
                };
                let delta = SeqScanOperator::new(
                    ctx.clone(),
                    table_id,
                    columns.clone(),
                    Some(full_predicate),
                    false,
                    None,
                )
                .await?
                .append_only();
                Some(delta)
            } else {
                None
            };

            // The B+tree yields entries in ascending key order. A descending
            // scan reads the same entries the other way, which is what lets
            // an ORDER BY ... DESC be answered without a sort. The list is
            // already materialized, so this is a reversal rather than a
            // second traversal
            if descending {
                locators.reverse();
            }

            // The range scan already ran to completion above, so the number
            // of entries it examined is known here and recorded once
            let io_stats = crate::operator::IndexScanStats::open(
                &ctx,
                table_id.0,
                index_entry.id.0,
                locators.len(),
            );

            return Ok(Self {
                index_state: Some(IndexScanState {
                    ctx,
                    table_entry,
                    output_columns: columns,
                    column_to_builder,
                    remaining_predicate: effective_remaining,
                    track_tuple_ids,
                    locators,
                    columnar,
                    cursor: 0,
                    branch_id,
                    finished: false,
                    io_stats,
                }),
                fallback: None,
                append_delta,
            });
        }

        // Fallback: no B+ tree instance available. Use sequential scan
        // with the full predicate as a post-filter.
        let combined = match remaining_predicate {
            Some(rest) => BoundExpr::BinaryOp {
                left: Box::new(predicate),
                op: BinaryOperator::And,
                right: Box::new(rest),
                type_id: zyron_common::TypeId::Boolean,
            },
            None => predicate,
        };

        let inner = SeqScanOperator::new(
            ctx,
            table_id,
            columns,
            Some(combined),
            track_tuple_ids,
            None,
        )
        .await?;

        Ok(Self {
            index_state: None,
            fallback: Some(inner),
            append_delta: None,
        })
    }
}

impl Operator for IndexScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        // Delegate to fallback sequential scan if no index state.
        if let Some(ref mut fallback) = self.fallback {
            return fallback.next();
        }

        Box::pin(async move {
            // Drain the main index path first; when it is exhausted, drain the
            // branch insert delta (present only for branch-active reads).
            if let Some(state) = self.index_state.as_mut() {
                if !state.finished {
                    if let Some(batch) = state.next_batch().await? {
                        return Ok(Some(batch));
                    }
                    state.finished = true;
                }
            }
            if let Some(delta) = self.append_delta.as_mut() {
                return delta.next().await;
            }
            Ok(None)
        })
    }
}

impl IndexScanState {
    /// Produces the next batch from the pre-collected index tuple ids, or None
    /// when they are exhausted. Each tuple id is resolved through the active
    /// branch's override chain before fetch so branch deletes and modifications
    /// (their cow page slot is tombstoned) drop out.
    async fn next_batch(&mut self) -> Result<Option<ExecutionBatch>> {
        self.ctx.check_cancelled()?;

        let batch_size = self.ctx.batch_size;
        let mut builders =
            scan_builders(&self.output_columns, &self.table_entry.columns, batch_size);
        let mut result_locators: Vec<zyron_common::RowLocator> = if self.track_tuple_ids {
            Vec::with_capacity(batch_size)
        } else {
            Vec::new()
        };
        let mut row_count: usize = 0;

        // Fetch rows using the pre-collected locators. Heap entries read
        // directly from the buffer pool frame's data via the read lock,
        // avoiding the 16KB stack copy + Box allocation that
        // read_page_through_pool would do per call. Concurrent atomic
        // inserts coordinate via the slot's AtomicU32 commit so our read
        // sees either uncommitted (length=0, skip) or committed bytes
        // consistently. Columnar entries were pre-fetched in one batched
        // pass with visibility applied
        // DML consumers route a batch to the heap or the columnar mutation
        // path as a whole, so tracked batches stay homogeneous per storage
        // kind, a kind change closes the batch and the next call continues
        let mut batch_kind: Option<u8> = None;
        // Heap pages fetched to resolve this batch's locators, folded into the
        // table counters once when the batch is done.
        let mut pages_read: u64 = 0;
        while row_count < batch_size && self.cursor < self.locators.len() {
            let loc = self.locators[self.cursor];
            if self.track_tuple_ids {
                let kind = match loc {
                    zyron_common::RowLocator::Heap { .. } => 0u8,
                    zyron_common::RowLocator::Columnar { .. } => 1,
                    zyron_common::RowLocator::Lake { .. } => 2,
                };
                match batch_kind {
                    None => batch_kind = Some(kind),
                    Some(k) if k != kind => break,
                    _ => {}
                }
            }
            self.cursor += 1;

            let visible = match loc {
                zyron_common::RowLocator::Heap { page, slot } => {
                    // Resolve to the branch-local page when the branch copied
                    // it; the slot id is preserved by the page copy.
                    let phys_page = self.ctx.resolve_branch_page(self.branch_id, page);
                    pages_read += 1;

                    let frame_present = self.ctx.buffer_pool.fetch_page(phys_page).is_some();
                    if !frame_present {
                        let disk_data = self.ctx.disk_manager.read_page(phys_page).await?;
                        self.ctx.buffer_pool.load_page(phys_page, &disk_data)?;
                        // load_page pinned, frame_present path's fetch_page also
                        // pinned, in both cases we have one extra pin to balance
                    }

                    let frame = self
                        .ctx
                        .buffer_pool
                        .fetch_page(phys_page)
                        .expect("just pinned this page");
                    self.ctx.buffer_pool.unpin_page(phys_page, false);

                    let visible = {
                        let guard = frame.read_data();
                        let slot_id = zyron_storage::SlotId(slot);
                        match HeapPage::get_tuple_view_from_slice(&**guard, slot_id) {
                            None => false,
                            Some(view) => {
                                if view.is_deleted()
                                    || !view.header.is_visible_to(&self.ctx.snapshot)
                                {
                                    false
                                } else {
                                    decode_tuple_into_builders(
                                        view.data,
                                        &self.table_entry.columns,
                                        &self.column_to_builder,
                                        &mut builders,
                                    );
                                    true
                                }
                            }
                        }
                    };
                    self.ctx.buffer_pool.unpin_page(phys_page, false);
                    visible
                }
                zyron_common::RowLocator::Columnar { file_id, sys_rowid } => {
                    match self
                        .columnar
                        .as_ref()
                        .and_then(|f| f.columnar_row(file_id, sys_rowid))
                    {
                        Some(vals) => {
                            for (b, v) in builders.iter_mut().zip(vals.iter()) {
                                b.push(v);
                            }
                            true
                        }
                        // superseded, invisible to this snapshot, or reclaimed
                        None => false,
                    }
                }
                zyron_common::RowLocator::Lake { .. } => false,
            };

            if visible {
                if self.track_tuple_ids {
                    result_locators.push(loc);
                }
                row_count += 1;
            }
        }

        self.io_stats
            .record_batch(row_count as u64, pages_read * PAGE_SIZE as u64);

        if row_count == 0 {
            return Ok(None);
        }

        let mut batch = finalize_builders(builders);
        // ENCRYPTED columns decrypt before anything reads the batch, so the
        // post-filter below evaluates over plaintext
        decrypt_encrypted_columns(
            &self.ctx,
            &mut batch,
            &self.table_entry.columns,
            &self.output_columns,
        )?;
        let batch = batch;

        // Apply remaining predicate as a post-filter.
        if let Some(ref pred) = self.remaining_predicate {
            let mask_col = evaluate(pred, &batch, &self.output_columns, &self.ctx.params)?;
            let mask = column_to_mask(&mask_col);
            let filtered = batch.filter(&mask);

            if self.track_tuple_ids {
                let filtered_ids: Vec<zyron_common::RowLocator> = mask
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(i, &keep)| {
                            if keep { Some(result_locators[i]) } else { None }
                        },
                    )
                    .collect();
                return Ok(Some(ExecutionBatch::with_locators(filtered, filtered_ids)));
            }

            return Ok(Some(ExecutionBatch::new(filtered)));
        }

        if self.track_tuple_ids {
            Ok(Some(ExecutionBatch::with_locators(batch, result_locators)))
        } else {
            Ok(Some(ExecutionBatch::new(batch)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every page is handed out exactly once, in runs of the chunk size
    /// with the last one clipped to the end, and the cursor answers None
    /// forever after
    #[test]
    fn page_claims_cover_the_range_once() {
        let claims = PageClaims::new(3, 30);
        let mut runs = Vec::new();
        while let Some(run) = claims.claim(8) {
            runs.push(run);
        }
        assert_eq!(runs, vec![(3, 11), (11, 19), (19, 27), (27, 30)]);
        assert_eq!(claims.claim(8), None);
        assert_eq!(claims.claim(1), None);
    }

    #[test]
    fn an_empty_range_claims_nothing() {
        let claims = PageClaims::new(5, 5);
        assert_eq!(claims.claim(8), None);
    }

    /// Two claimers over one cursor split the pages between them without
    /// either seeing a page the other took
    #[test]
    fn claims_from_several_workers_never_overlap() {
        let claims = Arc::new(PageClaims::new(0, 1000));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let claims = Arc::clone(&claims);
                std::thread::spawn(move || {
                    let mut mine = Vec::new();
                    while let Some((start, end)) = claims.claim(PARALLEL_SCAN_CLAIM_PAGES) {
                        mine.extend(start..end);
                    }
                    mine
                })
            })
            .collect();
        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("claimer"))
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..1000).collect::<Vec<u64>>());
    }

    #[test]
    fn worker_count_follows_the_work_floor() {
        assert_eq!(parallel_workers_for_pages(0), 1);
        assert_eq!(parallel_workers_for_pages(1), 1);
        assert_eq!(
            parallel_workers_for_pages(PARALLEL_SCAN_MIN_PAGES_PER_WORKER),
            1
        );
        assert_eq!(
            parallel_workers_for_pages(PARALLEL_SCAN_MIN_PAGES_PER_WORKER + 1),
            2
        );
        assert_eq!(parallel_workers_for_pages(400), 13);
    }
}

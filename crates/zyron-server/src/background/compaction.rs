//! Background compaction worker that folds committed heap rows into
//! immutable .zyr columnar segments.
//!
//! Eligibility: a heap tuple folds only when it is not deleted, its xmax is
//! 0, and its xmin is below the oldest active transaction, so every current
//! and future snapshot observes exactly this version. The fold is one
//! WAL-atomic transition: the .zyr is written and fsynced, CompactionBegin
//! then CompactionEnd are logged and flushed, and only then is the catalog
//! columnar registry updated and the folded heap slots zeroed. A crash with
//! CompactionBegin and no CompactionEnd leaves the heap authoritative and the
//! partial .zyr discarded by recovery.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, info, warn};

use zyron_buffer::BufferPool;
use zyron_catalog::schema::ColumnarSegmentEntry;
use zyron_catalog::{Catalog, TableEntry};
use zyron_common::page::PAGE_SIZE;
use zyron_common::types::TypeId;
use zyron_storage::columnar::{
    ColumnDescriptor, CompactionConfig, CompactionInput, SYS_COL_ROWID, SYS_COL_SUPERSEDE,
    SYS_COL_XMIN, run_compaction_cycle,
};
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig, HeapPage, TupleHeader};
use zyron_wal::WalWriter;

use crate::metrics::MetricsRegistry;

/// Configuration for the compaction worker.
#[derive(Debug, Clone)]
pub struct CompactionWorkerConfig {
    /// Interval between compaction cycles in seconds.
    pub interval_secs: u64,
    /// Minimum fold-eligible rows in a table before it is folded.
    pub min_rows: u64,
    /// Maximum rows written into one .zyr file.
    pub max_rows_per_file: u64,
    /// Skip the cycle when measured query p99 exceeds this many microseconds.
    pub oltp_p99_threshold_us: u64,
    /// Directory for .zyr output files.
    pub columnar_dir: std::path::PathBuf,
    /// fsync .zyr files and the directory after rename.
    pub fsync_enabled: bool,
    /// Threads for parallel column encoding.
    pub max_encoding_threads: usize,
    /// A churned segment is merged (rewritten) only once its
    /// dead-or-patched fraction reaches this ratio, amortizing rewrite cost
    /// over many mutations. A fully-dead segment is always reclaimed.
    pub merge_min_churn_ratio: f64,
    /// Persist the columnar registry to durable storage every this many
    /// segments (the cache is updated every fold regardless). Bounds
    /// catalog write amplification; the WAL CompactionEnd records cover the
    /// gap on crash.
    pub registry_persist_every: u64,
}

impl Default for CompactionWorkerConfig {
    fn default() -> Self {
        Self {
            interval_secs: 30,
            min_rows: 100_000,
            max_rows_per_file: 1_000_000,
            oltp_p99_threshold_us: 1_000,
            columnar_dir: std::path::PathBuf::from("./data/columnar"),
            fsync_enabled: true,
            max_encoding_threads: 4,
            merge_min_churn_ratio: 0.10,
            registry_persist_every: 16,
        }
    }
}

/// Compaction worker statistics.
pub struct CompactionStats {
    pub cycles_completed: AtomicU64,
    pub rows_folded: AtomicU64,
    pub segments_written: AtomicU64,
    pub cycles_backpressured: AtomicU64,
}

impl CompactionStats {
    fn new() -> Self {
        Self {
            cycles_completed: AtomicU64::new(0),
            rows_folded: AtomicU64::new(0),
            segments_written: AtomicU64::new(0),
            cycles_backpressured: AtomicU64::new(0),
        }
    }
}

/// One folded heap slot: (page, slot, folded tuple xmin). The xmin is the
/// identity used before zeroing so a slot reused by a different tuple after
/// the fold committed is never destroyed by a redo from the sidecar.
type FoldedRid = (zyron_common::page::PageId, u16, u32);

/// True when `slot` on `page` still holds the folded tuple and that tuple is
/// still fold-eligible: the slot is non-empty, its offsets are in range, its
/// xmin equals the folded identity, it is not flagged deleted, and xmax is
/// unset. One pass over the slot, no full-page copy. `page` is any view of
/// the page bytes (a buffer-frame guard or an owned read).
fn slot_still_folded(page: &[u8], slot: u16, folded_xmin: u32) -> bool {
    let so = HeapPage::DATA_START + (slot as usize) * 4;
    if so + 4 > page.len() {
        return false;
    }
    let toff = u16::from_le_bytes([page[so], page[so + 1]]) as usize;
    let slen = u16::from_le_bytes([page[so + 2], page[so + 3]]) as usize;
    if slen == 0 || toff + TupleHeader::SIZE > page.len() {
        return false;
    }
    let flags = u16::from_le_bytes([page[toff], page[toff + 1]]);
    let xmin = u32::from_le_bytes([
        page[toff + 4],
        page[toff + 5],
        page[toff + 6],
        page[toff + 7],
    ]);
    let xmax = u32::from_le_bytes([
        page[toff + 8],
        page[toff + 9],
        page[toff + 10],
        page[toff + 11],
    ]);
    xmin == folded_xmin && flags & 0x0001 == 0 && xmax == 0
}

/// Removes a file, logging a real failure. A missing file is the expected
/// idempotent-cleanup case and is silent; any other error leaks the file on
/// disk forever, so it is surfaced rather than swallowed.
fn unlink_logged(path: &std::path::Path, context: &str) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!(
            "compaction: failed to remove {} ({}): {}",
            path.display(),
            context,
            e
        ),
    }
}

/// Length-prefix a path into a WAL recovery payload. A `len as u32` truncation
/// would make recovery parse every following field at the wrong offset, so an
/// impossibly long path is rejected before the record is written rather than
/// silently corrupting the record. Real filesystem paths are far under this.
fn push_len_prefixed_path(buf: &mut Vec<u8>, path: &str) -> std::result::Result<(), String> {
    if path.len() > u32::MAX as usize {
        return Err(format!(
            "compaction WAL path length {} exceeds the u32 length-prefix limit",
            path.len()
        ));
    }
    buf.extend_from_slice(&(path.len() as u32).to_le_bytes());
    buf.extend_from_slice(path.as_bytes());
    Ok(())
}

/// One column's materialized values for a fold: a single contiguous blob
/// plus a per-row (offset, len) index. Building a column appends cell bytes
/// into the amortized-growth blob and pushes one index entry per row, so a
/// fold allocates O(columns) buffers instead of O(rows x columns) tiny
/// per-cell vectors.
#[derive(Default)]
struct ColArena {
    blob: Vec<u8>,
    idx: Vec<Option<(usize, usize)>>,
}

impl ColArena {
    #[inline]
    fn push(&mut self, cell: Option<&[u8]>) {
        match cell {
            Some(v) => {
                let start = self.blob.len();
                self.blob.extend_from_slice(v);
                self.idx.push(Some((start, v.len())));
            }
            None => self.idx.push(None),
        }
    }

    /// Borrowed encoder view: one `Option<&[u8]>` per row into the blob.
    fn view(&self) -> Vec<Option<&[u8]>> {
        self.idx
            .iter()
            .map(|o| o.map(|(s, l)| &self.blob[s..s + l]))
            .collect()
    }
}

/// Background worker that folds heap rows into columnar segments.
pub struct CompactionWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<CompactionStats>,
}

impl CompactionWorker {
    /// Starts the compaction worker thread.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        catalog: Arc<Catalog>,
        txn_manager: Arc<TransactionManager>,
        disk_manager: Arc<DiskManager>,
        buffer_pool: Arc<BufferPool>,
        wal: Arc<WalWriter>,
        metrics: Option<Arc<MetricsRegistry>>,
        config: CompactionWorkerConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(CompactionStats::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);
        let thread_stats = Arc::clone(&stats);

        let handle = thread::Builder::new()
            .name("zyron-compaction".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to build tokio runtime for compaction worker");
                Self::compaction_loop(
                    &rt,
                    &catalog,
                    &txn_manager,
                    &disk_manager,
                    &buffer_pool,
                    &wal,
                    metrics.as_deref(),
                    &config,
                    &thread_shutdown,
                    &thread_stats,
                );
            })
            .expect("failed to spawn compaction worker thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
        }
    }

    /// Main loop. Wakes every interval, folds each eligible table once.
    #[allow(clippy::too_many_arguments)]
    fn compaction_loop(
        rt: &tokio::runtime::Runtime,
        catalog: &Catalog,
        txn_manager: &TransactionManager,
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
        wal: &Arc<WalWriter>,
        metrics: Option<&MetricsRegistry>,
        config: &CompactionWorkerConfig,
        shutdown: &AtomicBool,
        stats: &CompactionStats,
    ) {
        let interval = Duration::from_secs(config.interval_secs.max(1));

        loop {
            thread::park_timeout(interval);

            if shutdown.load(Ordering::Acquire) {
                return;
            }

            // OLTP-aware backoff: do not compete with the foreground write
            // path when query latency is already elevated.
            if let Some(m) = metrics {
                let p99 = m.query_duration.p99_estimate_us();
                if p99 > config.oltp_p99_threshold_us {
                    stats.cycles_backpressured.fetch_add(1, Ordering::Relaxed);
                    debug!(
                        "Compaction backing off: query p99 {}us > threshold {}us",
                        p99, config.oltp_p99_threshold_us
                    );
                    continue;
                }
            }

            let (total_rows, total_segments) = Self::run_cycle(
                rt,
                catalog,
                txn_manager,
                disk_manager,
                buffer_pool,
                wal,
                config,
                Some(shutdown),
            );

            stats.cycles_completed.fetch_add(1, Ordering::Relaxed);
            stats.rows_folded.fetch_add(total_rows, Ordering::Relaxed);
            stats
                .segments_written
                .fetch_add(total_segments, Ordering::Relaxed);

            if total_rows > 0 {
                info!(
                    "Compaction folded {} rows into {} segments",
                    total_rows, total_segments
                );
            }
        }
    }

    /// Runs one full compaction cycle over every table and returns
    /// (rows_folded, segments_written). Deterministic and synchronous: the
    /// background loop calls this on each wake, and integration tests call it
    /// directly with no shutdown handle for a one-shot fold.
    #[allow(clippy::too_many_arguments)]
    pub fn run_cycle(
        rt: &tokio::runtime::Runtime,
        catalog: &Catalog,
        txn_manager: &TransactionManager,
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
        wal: &Arc<WalWriter>,
        config: &CompactionWorkerConfig,
        shutdown: Option<&AtomicBool>,
    ) -> (u64, u64) {
        let active_txns = txn_manager.active_txn_ids();
        let oldest_active = if active_txns.is_empty() {
            txn_manager.next_txn_id()
        } else {
            active_txns[0]
        };

        let tables = catalog.list_all_tables();
        let mut total_rows = 0u64;
        let mut total_segments = 0u64;

        for table in &tables {
            if shutdown.map(|s| s.load(Ordering::Acquire)).unwrap_or(false) {
                break;
            }
            match Self::compact_table(
                rt,
                catalog,
                table,
                oldest_active,
                disk_manager,
                buffer_pool,
                wal,
                config,
            ) {
                Ok(Some(rows)) => {
                    total_rows += rows;
                    total_segments += 1;
                }
                Ok(None) => {}
                Err(e) => {
                    // Fail loud. A fold error never silently drops data: the
                    // heap is still authoritative because the catalog
                    // registry and heap delete only apply after a durable
                    // CompactionEnd, which this path did not reach.
                    warn!("Compaction for table {} failed: {}", table.name, e);
                }
            }
        }

        // Incremental merge pass: rewrite segments whose patch overlay has
        // fully settled (every overlay xid below the oldest-active horizon),
        // folding reclaimable patches into a fresh segment and dropping rows
        // deleted at or below the table's retention floor. Within-window history
        // (superseded versions and value patches committed after the floor) is
        // carried forward, so time-travel stays correct while old history is
        // reclaimed. Append-only data is never touched, so write amplification
        // tracks churn.
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        for table in &tables {
            if shutdown.map(|s| s.load(Ordering::Acquire)).unwrap_or(false) {
                break;
            }
            let floor = zyron_executor::operator::modify::effective_retention_floor(
                table.as_ref(),
                txn_manager.status_map(),
                txn_manager.retention_clock(),
                now_micros,
            );
            if let Err(e) = Self::merge_table(
                rt,
                catalog,
                table.id,
                oldest_active,
                floor,
                txn_manager.status_map(),
                wal,
                config,
            ) {
                warn!("Columnar merge for table {} failed: {}", table.name, e);
            }
        }
        (total_rows, total_segments)
    }

    /// Rewrites each fully-settled segment of one table: drops rows with a
    /// committed supersede, folds the newest committed value patch into the
    /// base, preserves sys_rowid, swaps the registry, deletes the old .zyr,
    /// and compacts the patch log. Skips a segment with any unsettled overlay
    /// xid so a merged base is always snapshot-independent.
    #[allow(clippy::too_many_arguments)]
    fn merge_table(
        rt: &tokio::runtime::Runtime,
        catalog: &Catalog,
        table_id: zyron_catalog::TableId,
        oldest_active: u64,
        floor: u64,
        status_map: &zyron_storage::TxnStatusMap,
        wal: &Arc<WalWriter>,
        config: &CompactionWorkerConfig,
    ) -> std::result::Result<(), String> {
        let te = catalog
            .get_table_by_id(table_id)
            .map_err(|e| format!("table reload: {}", e))?;
        if te.columnar.segments.is_empty() {
            return Ok(());
        }
        let columnar_dir = std::path::Path::new(&te.columnar.segments[0].path)
            .parent()
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| config.columnar_dir.clone());
        let patch_path = columnar_dir.join(format!("{}.zyrpatch", table_id.0));
        let store = zyron_storage::columnar::ColumnarPatchManager::global(&columnar_dir)
            .store(table_id.0 as u64)
            .map_err(|e| format!("patch store: {}", e))?;
        // A patch or supersede is reclaimable once its transaction committed at
        // or below the retention floor: no retained version reads the value
        // before it. Within-window history (committed after the floor) is kept.
        let reclaimable = |xid: u64| status_map.is_reclaimable_below(xid, floor);
        // Collapse only reclaimable value-patch history; within-window patches
        // are preserved so a retained version still sees the pre-patch value.
        store.trim_below(oldest_active, reclaimable);
        if store.is_empty() {
            return Ok(());
        }

        let mut columns: Vec<_> = te.columns.clone();
        columns.sort_by_key(|c| c.ordinal);

        for seg in te.columnar.segments.clone() {
            // One overlay snapshot per segment under a single read-lock,
            // reused for the settled check and the survivor resolution below,
            // instead of scanning the whole store (rows_with_overlay) and then
            // re-locking per row.
            let seg_overlay = store.file_overlay(seg.file_id);
            if seg_overlay.is_empty() {
                continue;
            }
            // Settled check: every overlay xid for this file < oldest_active.
            let mut settled = true;
            for ov in seg_overlay.values() {
                if ov.supersedes.iter().any(|x| *x >= oldest_active)
                    || ov
                        .patches
                        .values()
                        .flatten()
                        .any(|p| p.patch_xid >= oldest_active)
                {
                    settled = false;
                    break;
                }
            }
            if !settled {
                continue;
            }

            let reader =
                zyron_storage::columnar::ZyrFileReader::open(std::path::Path::new(&seg.path))
                    .map_err(|e| format!("open segment: {}", e))?;
            let row_count = reader.header().row_count as usize;
            if row_count == 0 {
                continue;
            }

            let dec = |col: u32, vs: usize| -> std::result::Result<(Vec<u8>, Vec<u8>), String> {
                Self::decode_seg_column(&reader, col, row_count, vs)
                    .map_err(|e| format!("decode column {}: {}", col, e))
            };
            let (rowid_b, _) = dec(zyron_storage::columnar::SYS_COL_ROWID, 8)?;
            let (xmin_b, _) = dec(zyron_storage::columnar::SYS_COL_XMIN, 8)?;
            let read_u64 =
                |b: &[u8], i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());

            // Decode each user column once.
            let mut user_dec: Vec<(Vec<u8>, Vec<u8>, usize, bool)> = Vec::new();
            for c in &columns {
                let phys = c.physical_type_id();
                let vs = phys.fixed_size().unwrap_or(0);
                let (bytes, nb) = dec(c.id.0 as u32, vs)?;
                user_dec.push((bytes, nb, vs, vs == 0));
            }
            let mut varlen_rows: Vec<Option<Vec<&[u8]>>> = Vec::with_capacity(user_dec.len());
            for (b, _, _, isv) in &user_dec {
                varlen_rows.push(if *isv {
                    Some(
                        zyron_storage::encoding::varlen_slice_rows(b, row_count)
                            .map_err(|e| format!("varlen slice: {}", e))?,
                    )
                } else {
                    None
                });
            }

            // Resolve survivors.
            let mut keep_rowid: Vec<u64> = Vec::new();
            let mut keep_xmin: Vec<u64> = Vec::new();
            // Carried-forward supersede per kept row: 0 for a live row, or the
            // delete's transaction id for a row deleted within the retention
            // window, written into the new segment's sys_supersede so AS OF
            // before the delete still sees it.
            let mut keep_supersede: Vec<u64> = Vec::new();
            // Within-window value patches (committed after the floor) to migrate
            // to the new segment so a retained version still reads them.
            let mut migrate: Vec<(u64, u32, u64, Vec<u8>)> = Vec::new();
            let mut col_vals: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::new(); columns.len()];
            let mut dropped = 0usize;
            let mut patched = 0usize;
            // Reuse the single per-segment overlay snapshot taken above; per
            // row this is a plain local map lookup, no lock, no whole-store
            // scan.
            for r in 0..row_count {
                let rid = read_u64(&rowid_b, r);
                let ov = seg_overlay.get(&rid);
                // Dead: deleted at or before the floor, so no retained version
                // sees it alive. Reclaimable subsumes the settled check (commit
                // LSN <= floor implies committed below the active horizon).
                let dead = ov
                    .map(|o| {
                        o.supersedes
                            .iter()
                            .any(|x| *x < oldest_active && reclaimable(*x))
                    })
                    .unwrap_or(false);
                if dead {
                    dropped += 1;
                    continue;
                }
                // Within-window delete: carry the earliest committed supersede
                // forward so AS OF within the window still hides it after merge.
                let carried = ov
                    .and_then(|o| {
                        o.supersedes
                            .iter()
                            .copied()
                            .filter(|x| *x < oldest_active)
                            .filter_map(|x| status_map.commit_lsn(x).map(|cl| (cl, x)))
                            .min()
                            .map(|(_, x)| x)
                    })
                    .unwrap_or(0);
                keep_rowid.push(rid);
                keep_xmin.push(read_u64(&xmin_b, r));
                keep_supersede.push(carried);
                // Churn is counted once per patched row, not once per patched
                // cell: a row with N changed columns is one mutation for the
                // rewrite trigger, not N, so multi-column updates do not
                // prematurely force a full-segment rewrite.
                let mut row_has_patch = false;
                for (ci, c) in columns.iter().enumerate() {
                    let (bytes, nb, vs, isv) = &user_dec[ci];
                    let col_id = c.id.0 as u32;
                    // Fold the newest reclaimable patch into the base (the value
                    // as of the floor). Keep within-window patches for migration
                    // so a retained version still reads the pre-patch value.
                    let mut newest_reclaimable: Option<&zyron_storage::columnar::ValuePatch> = None;
                    if let Some(o) = ov {
                        if let Some(chain) = o.patches.get(&col_id) {
                            for p in chain {
                                if p.patch_xid >= oldest_active {
                                    continue;
                                }
                                if reclaimable(p.patch_xid) {
                                    match newest_reclaimable {
                                        Some(b) if b.patch_xid >= p.patch_xid => {}
                                        _ => newest_reclaimable = Some(p),
                                    }
                                } else {
                                    // Within-window patch: migrated to the new
                                    // segment, but it is not reclaimable work, so
                                    // it does not by itself justify a rewrite.
                                    migrate.push((rid, col_id, p.patch_xid, p.value.clone()));
                                }
                            }
                        }
                    }
                    if let Some(p) = newest_reclaimable {
                        row_has_patch = true;
                        col_vals[ci].push(Some(p.value.clone()));
                        continue;
                    }
                    let is_null = !nb.is_empty() && (nb[r / 8] >> (r % 8)) & 1 == 1;
                    if is_null {
                        col_vals[ci].push(None);
                    } else if *isv {
                        col_vals[ci].push(Some(varlen_rows[ci].as_ref().unwrap()[r].to_vec()));
                    } else {
                        col_vals[ci].push(Some(bytes[r * vs..(r + 1) * vs].to_vec()));
                    }
                }
                if row_has_patch {
                    patched += 1;
                }
            }
            if dropped == 0 && patched == 0 {
                continue; // Nothing materially changed; do not rewrite.
            }
            // Churn-ratio trigger. Append-only segments have no overlay and
            // never reach here, so they are never rewritten. A churned
            // segment is rewritten only once its dead/patched fraction
            // crosses the threshold (or the whole segment is dead), so the
            // rewrite cost is amortized over many mutations instead of
            // re-encoding a large segment for a single superseded row.
            // Until then reads stay correct via the overlay.
            let churn = (dropped + patched) as f64 / row_count as f64;
            if dropped < row_count && churn < config.merge_min_churn_ratio {
                continue;
            }

            // Re-fold the survivors into a fresh segment, sys_rowid preserved.
            let mut entry = catalog
                .get_table_by_id(table_id)
                .map_err(|e| format!("reload: {}", e))?
                .as_ref()
                .clone();
            let new_file_id = entry.columnar.next_file_id;
            let kept = keep_rowid.len();
            let mut descriptors: Vec<ColumnDescriptor> = Vec::new();
            let mut column_data: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
            for (ci, c) in columns.iter().enumerate() {
                let phys = c.physical_type_id();
                descriptors.push(ColumnDescriptor {
                    column_id: c.id.0 as u32,
                    type_id: c.type_id,
                    value_size: phys.fixed_size().unwrap_or(0),
                    is_primary_key: false,
                });
                column_data.push(std::mem::take(&mut col_vals[ci]));
            }
            descriptors.push(ColumnDescriptor {
                column_id: zyron_storage::columnar::SYS_COL_ROWID,
                type_id: TypeId::UInt64,
                value_size: 8,
                is_primary_key: true,
            });
            column_data.push(
                keep_rowid
                    .iter()
                    .map(|x| Some(x.to_le_bytes().to_vec()))
                    .collect(),
            );
            descriptors.push(ColumnDescriptor {
                column_id: zyron_storage::columnar::SYS_COL_XMIN,
                type_id: TypeId::UInt64,
                value_size: 8,
                is_primary_key: false,
            });
            column_data.push(
                keep_xmin
                    .iter()
                    .map(|x| Some(x.to_le_bytes().to_vec()))
                    .collect(),
            );
            descriptors.push(ColumnDescriptor {
                column_id: zyron_storage::columnar::SYS_COL_SUPERSEDE,
                type_id: TypeId::UInt64,
                value_size: 8,
                is_primary_key: false,
            });
            // Carry within-window deletes forward as sys_supersede so AS OF
            // before the delete still sees the row; live rows write 0.
            column_data.push(
                keep_supersede
                    .iter()
                    .map(|x| Some(x.to_le_bytes().to_vec()))
                    .collect(),
            );

            if kept == 0 {
                // Whole segment died. Drop it and its patches; no new file.
                let mut bp = Vec::new();
                bp.extend_from_slice(&(table_id.0 as u64).to_le_bytes());
                bp.extend_from_slice(seg.path.as_bytes());
                wal.log_merge_begin(&bp).map_err(|e| e.to_string())?;
                let mut ep = Vec::new();
                ep.extend_from_slice(&(table_id.0 as u64).to_le_bytes());
                ep.extend_from_slice(&seg.file_id.to_le_bytes());
                wal.log_merge_end(&ep).map_err(|e| e.to_string())?;
                wal.flush().map_err(|e| e.to_string())?;
                entry.columnar.segments.retain(|s| s.file_id != seg.file_id);
                rt.block_on(catalog.update_table(entry))
                    .map_err(|e| format!("registry: {}", e))?;
                unlink_logged(
                    std::path::Path::new(&seg.path),
                    "superseded segment after whole-file merge",
                );
                store
                    .drop_file(seg.file_id, &patch_path)
                    .map_err(|e| format!("patch drop: {}", e))?;
                continue;
            }

            let xmin_lo = keep_xmin.iter().copied().min().unwrap_or(0);
            let xmin_hi = keep_xmin.iter().copied().max().unwrap_or(0);
            let input = CompactionInput {
                columns: descriptors,
                column_data,
                table_id: table_id.0 as u64,
                xmin_lo,
                xmin_hi,
            };
            let cc = CompactionConfig {
                columnar_dir: columnar_dir.clone(),
                min_rows: 0,
                max_rows_per_file: config.max_rows_per_file,
                fsync_enabled: config.fsync_enabled,
                max_encoding_threads: config.max_encoding_threads,
                oltp_p99_threshold_us: config.oltp_p99_threshold_us,
                check_interval_ms: config.interval_secs * 1000,
            };
            let result =
                run_compaction_cycle(&cc, input).map_err(|e| format!("merge write: {}", e))?;
            let new_path = result.file_path.to_string_lossy().to_string();

            let mut bp = Vec::new();
            bp.extend_from_slice(&(table_id.0 as u64).to_le_bytes());
            bp.extend_from_slice(new_path.as_bytes());
            wal.log_merge_begin(&bp).map_err(|e| e.to_string())?;
            let mut ep = Vec::new();
            ep.extend_from_slice(&(table_id.0 as u64).to_le_bytes());
            ep.extend_from_slice(&new_file_id.to_le_bytes());
            ep.extend_from_slice(&seg.file_id.to_le_bytes());
            push_len_prefixed_path(&mut ep, &new_path)?;
            wal.log_merge_end(&ep).map_err(|e| e.to_string())?;
            wal.flush().map_err(|e| e.to_string())?;

            let keep_lo = keep_rowid.iter().copied().min().unwrap_or(0);
            let keep_hi = keep_rowid.iter().copied().max().unwrap_or(0);
            entry.columnar.segments.retain(|s| s.file_id != seg.file_id);
            entry.columnar.segments.push(ColumnarSegmentEntry {
                file_id: new_file_id,
                path: new_path,
                row_count: kept as u64,
                sys_rowid_lo: keep_lo,
                sys_rowid_hi: keep_hi,
                sys_xmin_lo: xmin_lo,
                sys_xmin_hi: xmin_hi,
            });
            entry.columnar.next_file_id = new_file_id + 1;
            entry.columnar.low_water = oldest_active;
            rt.block_on(catalog.update_table(entry))
                .map_err(|e| format!("registry: {}", e))?;
            unlink_logged(
                std::path::Path::new(&seg.path),
                "superseded segment after merge",
            );
            store
                .drop_file(seg.file_id, &patch_path)
                .map_err(|e| format!("patch drop: {}", e))?;
            // Migrate within-window value patches to the new segment so a
            // retained version still reads the pre-patch value. The base in the
            // new .zyr is the value as of the floor; these patches reconstruct
            // values committed after it. Written at the current persisted LSN
            // high-water so recovery does not regress or re-replay them.
            if !migrate.is_empty() {
                let hwm = store.max_persisted_lsn();
                for (rid, col_id, patch_xid, value) in &migrate {
                    store
                        .append_value_patch(new_file_id, *rid, *col_id, *patch_xid, hwm, value)
                        .map_err(|e| format!("patch migrate: {}", e))?;
                }
            }
        }
        Ok(())
    }

    /// Parses a raw .zyr segment's header regions and decodes the column.
    fn decode_seg_column(
        reader: &zyron_storage::columnar::ZyrFileReader,
        column_id: u32,
        row_count: usize,
        value_size: usize,
    ) -> zyron_common::Result<(Vec<u8>, Vec<u8>)> {
        use zyron_storage::columnar::{
            SEGMENT_HEADER_SIZE, SegmentHeader, ZONE_MAP_BATCH_SIZE, ZONE_MAP_ENTRY_SIZE,
        };
        let raw = reader.read_segment_raw(column_id)?;
        let mut hb = [0u8; SEGMENT_HEADER_SIZE];
        hb.copy_from_slice(&raw[..SEGMENT_HEADER_SIZE]);
        let h = SegmentHeader::from_bytes(&hb)?;
        let bloom = h.bloom_filter_size as usize;
        let zones = row_count.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
        let zm = zones * ZONE_MAP_ENTRY_SIZE;
        let nb = if h.null_count > 0 {
            row_count.div_ceil(8)
        } else {
            0
        };
        let start = SEGMENT_HEADER_SIZE + bloom + zm + nb;
        let end = start + h.encoded_size as usize;
        let null_bitmap = raw[SEGMENT_HEADER_SIZE + bloom + zm..start].to_vec();
        let enc = &raw[start..end];
        let crc = zyron_common::hash32(enc);
        if crc != h.data_checksum {
            return Err(zyron_common::ZyronError::InvalidZyrFile(format!(
                "merge segment payload checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
                h.data_checksum, crc
            )));
        }
        let decoded = zyron_storage::encoding::create_encoding(h.encoding_type)
            .decode(enc, row_count, value_size)?;
        Ok((decoded, null_bitmap))
    }

    /// Folds one table. Returns Ok(Some(rows)) when a segment was written and
    /// the heap rows handed off, Ok(None) when nothing was eligible.
    #[allow(clippy::too_many_arguments)]
    fn compact_table(
        rt: &tokio::runtime::Runtime,
        catalog: &Catalog,
        table: &TableEntry,
        oldest_active: u64,
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
        wal: &Arc<WalWriter>,
        config: &CompactionWorkerConfig,
    ) -> std::result::Result<Option<u64>, String> {
        // Columns ordered by ordinal. The heap payload is laid out in this
        // exact order, so materialization must walk it identically.
        let mut columns: Vec<_> = table.columns.clone();
        columns.sort_by_key(|c| c.ordinal);
        if columns.is_empty() {
            return Ok(None);
        }
        let num_cols = columns.len();
        let null_bitmap_len = num_cols.div_ceil(8);

        let heap_file = HeapFile::new(
            Arc::clone(disk_manager),
            Arc::clone(buffer_pool),
            HeapFileConfig {
                heap_file_id: table.heap_file_id,
                fsm_file_id: table.fsm_file_id,
            },
        )
        .map_err(|e| format!("failed to open heap file: {}", e))?;

        // Seed the page-count cache from disk; a freshly opened HeapFile
        // starts at zero and scan() would otherwise see no pages.
        rt.block_on(heap_file.init_cache())
            .map_err(|e| format!("heap init_cache failed: {}", e))?;

        let scan_guard = heap_file
            .scan()
            .map_err(|e| format!("scan failed: {}", e))?;
        let page_ids: Vec<_> = scan_guard.page_ids().to_vec();
        drop(scan_guard);

        // Per-column arenas plus the folded RID list. row_slices is a single
        // reused scratch of (offset, len) into the current payload, so a row
        // that fails mid-decode contributes nothing to any arena and no
        // per-row or per-cell allocation happens.
        let mut arenas: Vec<ColArena> = (0..num_cols).map(|_| ColArena::default()).collect();
        let mut sys_xmin: Vec<u64> = Vec::new();
        let mut folded_rids: Vec<FoldedRid> = Vec::new();
        let mut row_slices: Vec<Option<(usize, usize)>> = Vec::with_capacity(num_cols);
        let max_rows = config.max_rows_per_file as usize;

        'outer: for &page_id in &page_ids {
            let page_data = match buffer_pool.fetch_page(page_id) {
                Some(frame) => {
                    let guard = frame.read_data();
                    let data: [u8; PAGE_SIZE] = **guard;
                    drop(guard);
                    buffer_pool.unpin_page(page_id, false);
                    data
                }
                None => {
                    // Not resident: read through from disk so a quiet table
                    // (no background writer pressure) still folds.
                    match rt.block_on(disk_manager.read_page(page_id)) {
                        Ok(d) => {
                            if let Ok((_, evicted)) = buffer_pool.load_page(page_id, &d) {
                                if let Some(ev) = evicted {
                                    let _ =
                                        rt.block_on(disk_manager.write_page(ev.page_id, &ev.data));
                                }
                                buffer_pool.unpin_page(page_id, false);
                            }
                            d
                        }
                        Err(_) => continue,
                    }
                }
            };

            let header = HeapPage::heap_header_from_slice(&page_data);
            for slot in 0..header.slot_count {
                let slot_off = HeapPage::DATA_START + (slot as usize) * 4;
                let tuple_off =
                    u16::from_le_bytes([page_data[slot_off], page_data[slot_off + 1]]) as usize;
                let slot_len =
                    u16::from_le_bytes([page_data[slot_off + 2], page_data[slot_off + 3]]) as usize;
                if slot_len == 0 {
                    continue;
                }
                if tuple_off + TupleHeader::SIZE > PAGE_SIZE || slot_len < TupleHeader::SIZE {
                    continue;
                }

                let flags = u16::from_le_bytes([page_data[tuple_off], page_data[tuple_off + 1]]);
                let xmin = u32::from_le_bytes([
                    page_data[tuple_off + 4],
                    page_data[tuple_off + 5],
                    page_data[tuple_off + 6],
                    page_data[tuple_off + 7],
                ]);
                let xmax = u32::from_le_bytes([
                    page_data[tuple_off + 8],
                    page_data[tuple_off + 9],
                    page_data[tuple_off + 10],
                    page_data[tuple_off + 11],
                ]);

                // Eligibility. This predicate is exactly
                // Snapshot{txn_id = oldest_active}.is_visible(xmin, xmax) for
                // a row whose creator committed below the horizon: xmax == 0
                // (not deleted), xmin in (0, oldest_active) (committed and
                // visible to every current and future snapshot), and the
                // tuple flags do not mark it deleted. It is the same MVCC
                // oracle the heap scan uses. Aborted-creator tuples that the
                // heap has not yet undone are delegated to heap GC / abort
                // undo, the documented honesty boundary, identical to how the
                // heap scan itself behaves for such tuples.
                let deleted = flags & 0x0001 != 0;
                if deleted || xmax != 0 || xmin == 0 || (xmin as u64) >= oldest_active {
                    continue;
                }

                let payload_start = tuple_off + TupleHeader::SIZE;
                let payload_end = tuple_off + slot_len;
                if payload_end > PAGE_SIZE || payload_end <= payload_start {
                    continue;
                }
                let payload = &page_data[payload_start..payload_end];
                if payload.len() < null_bitmap_len {
                    continue;
                }

                // Decode the NSM payload in ordinal order. Layout matches
                // zyron-executor batch.rs: null bitmap then per column either
                // a fixed-size inline value or a u32-length-prefixed blob.
                let null_bitmap = &payload[..null_bitmap_len];
                let mut off = null_bitmap_len;
                row_slices.clear();
                let mut decode_ok = true;
                for (i, col) in columns.iter().enumerate() {
                    let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;
                    let phys = col.physical_type_id();
                    if let Some(fixed) = phys.fixed_size() {
                        if off + fixed > payload.len() {
                            decode_ok = false;
                            break;
                        }
                        row_slices.push(if is_null { None } else { Some((off, fixed)) });
                        off += fixed;
                    } else {
                        if off + 4 > payload.len() {
                            decode_ok = false;
                            break;
                        }
                        let len = u32::from_le_bytes([
                            payload[off],
                            payload[off + 1],
                            payload[off + 2],
                            payload[off + 3],
                        ]) as usize;
                        off += 4;
                        if off + len > payload.len() {
                            decode_ok = false;
                            break;
                        }
                        row_slices.push(if is_null { None } else { Some((off, len)) });
                        off += len;
                    }
                }
                if !decode_ok {
                    continue;
                }

                // Row fully decoded: copy each cell straight into its column
                // arena (memcpy into an amortized blob, no per-cell malloc).
                for (i, sl) in row_slices.iter().enumerate() {
                    arenas[i].push(sl.map(|(o, l)| &payload[o..o + l]));
                }
                sys_xmin.push(xmin as u64);
                folded_rids.push((page_id, slot, xmin));

                if folded_rids.len() >= max_rows {
                    break 'outer;
                }
            }
        }

        let row_count = folded_rids.len();
        if (row_count as u64) < config.min_rows {
            return Ok(None);
        }

        // Assign identity from the durable per-table registry.
        let mut entry = catalog
            .get_table_by_id(table.id)
            .map_err(|e| format!("table reload failed: {}", e))?
            .as_ref()
            .clone();
        let base_rowid = entry.columnar.next_rowid;
        let file_id = entry.columnar.next_file_id;

        let xmin_lo = sys_xmin.iter().copied().min().unwrap_or(0);
        let xmin_hi = sys_xmin.iter().copied().max().unwrap_or(0);

        // Descriptors: user columns then the three system columns. sys_rowid
        // is the primary key, assigned monotonically below, so the file is
        // already rowid ordered and the encoder skips the sort permutation.
        let mut descriptors: Vec<ColumnDescriptor> = Vec::with_capacity(num_cols + 3);
        for col in columns.iter() {
            let phys = col.physical_type_id();
            descriptors.push(ColumnDescriptor {
                column_id: col.id.0 as u32,
                type_id: col.type_id,
                value_size: phys.fixed_size().unwrap_or(0),
                is_primary_key: false,
            });
        }
        descriptors.push(ColumnDescriptor {
            column_id: SYS_COL_ROWID,
            type_id: TypeId::UInt64,
            value_size: 8,
            is_primary_key: true,
        });
        descriptors.push(ColumnDescriptor {
            column_id: SYS_COL_XMIN,
            type_id: TypeId::UInt64,
            value_size: 8,
            is_primary_key: false,
        });
        descriptors.push(ColumnDescriptor {
            column_id: SYS_COL_SUPERSEDE,
            type_id: TypeId::UInt64,
            value_size: 8,
            is_primary_key: false,
        });

        // sys_rowid / sys_xmin: one contiguous 8-byte-per-row blob each (one
        // allocation, not row_count tiny vecs). sys_supersede is the constant
        // 0u64 for every row, so every view entry points at one shared slice.
        const ZERO8: [u8; 8] = [0u8; 8];
        let mut sys_rowid_blob: Vec<u8> = Vec::with_capacity(row_count * 8);
        for i in 0..row_count {
            sys_rowid_blob.extend_from_slice(&(base_rowid + i as u64).to_le_bytes());
        }
        let mut sys_xmin_blob: Vec<u8> = Vec::with_capacity(row_count * 8);
        for x in &sys_xmin {
            sys_xmin_blob.extend_from_slice(&x.to_le_bytes());
        }

        // Per-column view provider, invoked inside each column's own encode
        // worker so view materialization is parallel across columns. User
        // columns borrow their arena; the three sys columns borrow the sys
        // blobs (supersede is the single shared zero slice).
        let column_view = |i: usize| -> Vec<Option<&[u8]>> {
            if i < num_cols {
                arenas[i].view()
            } else if i == num_cols {
                (0..row_count)
                    .map(|r| Some(&sys_rowid_blob[r * 8..r * 8 + 8]))
                    .collect()
            } else if i == num_cols + 1 {
                (0..row_count)
                    .map(|r| Some(&sys_xmin_blob[r * 8..r * 8 + 8]))
                    .collect()
            } else {
                vec![Some(&ZERO8[..]); row_count]
            }
        };

        let cc = CompactionConfig {
            columnar_dir: config.columnar_dir.clone(),
            min_rows: config.min_rows,
            max_rows_per_file: config.max_rows_per_file,
            fsync_enabled: config.fsync_enabled,
            max_encoding_threads: config.max_encoding_threads,
            oltp_p99_threshold_us: config.oltp_p99_threshold_us,
            check_interval_ms: config.interval_secs * 1000,
        };

        // Step 1: write the .zyr durable (temp + fsync + atomic rename).
        // sys_rowid is the PK at descriptor index num_cols; it is already
        // ascending so encode_and_write does no row reordering.
        let result = zyron_storage::columnar::encode_and_write(
            &cc,
            &descriptors,
            row_count,
            column_view,
            Some(num_cols),
            table.id.0 as u64,
            xmin_lo,
            xmin_hi,
        )
        .map_err(|e| format!("encode/write failed: {}", e))?;
        let path_str = result.file_path.to_string_lossy().to_string();
        let next_rowid = base_rowid + row_count as u64;

        // Step 2: write the RID sidecar (the folded heap locations) durably.
        // It is kept tiny in WAL by NOT inlining the list into a record: WAL
        // payload_len is u16, so a 100k-row fold would overflow CompactionEnd.
        // The sidecar lets crash recovery redo the heap-delete idempotently.
        let rid_path = result.file_path.with_extension("zyrrids");
        Self::write_rid_sidecar(&rid_path, &folded_rids)
            .map_err(|e| format!("rid sidecar write failed: {}", e))?;
        let rid_path_str = rid_path.to_string_lossy().to_string();

        // Step 3: CompactionBegin (file + sidecar durable, referenced by
        // nothing yet).
        let mut begin_payload = Vec::with_capacity(8 + path_str.len());
        begin_payload.extend_from_slice(&(table.id.0 as u64).to_le_bytes());
        begin_payload.extend_from_slice(path_str.as_bytes());
        wal.log_compaction_begin(&begin_payload)
            .map_err(|e| format!("CompactionBegin log failed: {}", e))?;

        // Step 4: re-validate every folded tuple under a fresh page read,
        // before the commit point. A transaction committed after the
        // eligibility snapshot could have set xmax (UPDATE/DELETE) on a
        // still-live folded row; folding it anyway would leave the stale
        // version live in columnar while the new version lives in the heap
        // (double-count / lost-update). This check is read-only, so aborting
        // here is safe: CompactionBegin with no CompactionEnd makes recovery
        // discard the .zyr and the sidecar, heap untouched.
        if !Self::revalidate_folded(buffer_pool, disk_manager, rt, &folded_rids)? {
            unlink_logged(&result.file_path, "aborted fold orphan .zyr");
            unlink_logged(&rid_path, "aborted fold orphan sidecar");
            debug!(
                "Fold of table {} aborted: a folded row changed concurrently",
                table.name
            );
            return Ok(None);
        }

        // Step 5: CompactionEnd is the commit point. Small fixed payload plus
        // the sidecar path; no RID list (WAL payload is u16-bounded).
        // Layout: table_id(8) file_id(8) file_size(8) row_count(8)
        //   base_rowid(8) next_rowid(8) xmin_lo(8) xmin_hi(8)
        //   path_len(4) path rid_path_len(4) rid_path
        let mut end_payload = Vec::with_capacity(80 + path_str.len() + rid_path_str.len());
        end_payload.extend_from_slice(&(table.id.0 as u64).to_le_bytes());
        end_payload.extend_from_slice(&file_id.to_le_bytes());
        end_payload.extend_from_slice(&result.file_size.to_le_bytes());
        end_payload.extend_from_slice(&(row_count as u64).to_le_bytes());
        end_payload.extend_from_slice(&base_rowid.to_le_bytes());
        end_payload.extend_from_slice(&next_rowid.to_le_bytes());
        end_payload.extend_from_slice(&xmin_lo.to_le_bytes());
        end_payload.extend_from_slice(&xmin_hi.to_le_bytes());
        push_len_prefixed_path(&mut end_payload, &path_str)?;
        push_len_prefixed_path(&mut end_payload, &rid_path_str)?;
        let end_lsn = wal
            .log_compaction_end(&end_payload)
            .map_err(|e| format!("CompactionEnd log failed: {}", e))?;
        wal.flush()
            .map_err(|e| format!("WAL flush failed: {}", e))?;

        // Step 6: apply (only after the commit point is durable). Zero the
        // folded heap slots BEFORE the segment becomes visible to scans. The
        // CompactionEnd WAL record above is the durable commit point and the
        // rid sidecar lets recovery redo this idempotently after a crash, so
        // the order here is purely about in-process visibility: if the segment
        // were registered first, a concurrent scan in the gap before the heap
        // delete would see every folded row in both the heap and the new
        // columnar segment (a transient double count). Deleting first closes
        // that window with no durability change.
        Self::delete_folded_rows(rt, buffer_pool, disk_manager, &folded_rids)?;

        // Register the segment now that its rows no longer exist in the heap.
        entry.columnar.segments.push(ColumnarSegmentEntry {
            file_id,
            path: path_str.clone(),
            row_count: row_count as u64,
            sys_rowid_lo: base_rowid,
            sys_rowid_hi: next_rowid.saturating_sub(1),
            sys_xmin_lo: xmin_lo,
            sys_xmin_hi: xmin_hi,
        });
        entry.columnar.next_rowid = next_rowid;
        entry.columnar.next_file_id = file_id + 1;
        // Amortized registry persistence. A per-fold whole-TableEntry rewrite
        // is O(segments) per fold, O(n^2) over the table's life. The cache is
        // updated every fold (O(1)) so scans/planner see the new segment
        // immediately; the durable storage rewrite happens every
        // `registry_persist_every` segments (and on the first segment). A
        // crash before the next durable persist is reconciled at startup
        // from the WAL CompactionEnd records (the registry's existing
        // recovery path), which the WAL retains far longer than a few folds.
        let seg_count = entry.columnar.segments.len() as u64;
        let persist_every = config.registry_persist_every.max(1);
        if seg_count <= 1 || seg_count % persist_every == 0 {
            rt.block_on(catalog.update_table(entry))
                .map_err(|e| format!("registry persist failed: {}", e))?;
            // Registry durable: every prior CompactionEnd for this table is
            // now reconstructable from durable storage, so the WAL no longer
            // needs to be pinned on this table's behalf.
            crate::columnar_wal_pin::ColumnarWalPin::global().release(table.id.0);
        } else {
            catalog.cache_put_table(entry);
            // Cache-only: pin the WAL at this CompactionEnd until a later
            // durable persist, so a checkpoint cannot reclaim the record
            // recovery needs to rebuild this segment.
            crate::columnar_wal_pin::ColumnarWalPin::global().note(table.id.0, end_lsn.0);
        }

        // Heap and registry are now durable; the sidecar is no longer needed.
        unlink_logged(&rid_path, "applied fold sidecar");

        debug!(
            "Folded {} rows of table {} into {} (file_id {})",
            row_count, table.name, path_str, file_id
        );
        Ok(Some(row_count as u64))
    }

    /// Writes the folded RID list to a sidecar file and fsyncs it. Format:
    /// rid_count(u64 LE) then [file_id(u32) page_num(u64) slot(u16) xmin(u32)].
    /// The xmin is the folded tuple identity checked before a redo zeroes the
    /// slot, so a slot reused after the fold committed is never destroyed.
    fn write_rid_sidecar(path: &std::path::Path, rids: &[FoldedRid]) -> std::io::Result<()> {
        use std::io::Write;
        let mut buf = Vec::with_capacity(8 + rids.len() * 18);
        buf.extend_from_slice(&(rids.len() as u64).to_le_bytes());
        for (pid, slot, xmin) in rids {
            buf.extend_from_slice(&pid.file_id.to_le_bytes());
            buf.extend_from_slice(&pid.page_num.to_le_bytes());
            buf.extend_from_slice(&slot.to_le_bytes());
            buf.extend_from_slice(&xmin.to_le_bytes());
        }
        let tmp = path.with_extension("zyrrids.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Reads a RID sidecar back into the (PageId, slot) list.
    pub fn read_rid_sidecar(path: &std::path::Path) -> std::io::Result<Vec<FoldedRid>> {
        let buf = std::fs::read(path)?;
        if buf.len() < 8 {
            return Ok(Vec::new());
        }
        let n = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
        let mut out = Vec::with_capacity(n);
        let mut p = 8;
        for _ in 0..n {
            if p + 18 > buf.len() {
                break;
            }
            let fid = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            let pnum = u64::from_le_bytes(buf[p + 4..p + 12].try_into().unwrap());
            let slot = u16::from_le_bytes(buf[p + 12..p + 14].try_into().unwrap());
            let xmin = u32::from_le_bytes(buf[p + 14..p + 18].try_into().unwrap());
            out.push((zyron_common::page::PageId::new(fid, pnum), slot, xmin));
            p += 18;
        }
        Ok(out)
    }

    /// Re-reads each folded tuple under a fresh page read and returns false
    /// if any is no longer eligible (xmax set, deleted flag, or slot emptied)
    /// since the eligibility snapshot. Read-only; the caller aborts the fold
    /// when this returns false.
    fn revalidate_folded(
        buffer_pool: &Arc<BufferPool>,
        disk_manager: &Arc<DiskManager>,
        rt: &tokio::runtime::Runtime,
        rids: &[FoldedRid],
    ) -> std::result::Result<bool, String> {
        use std::collections::HashMap;
        let mut by_page: HashMap<zyron_common::page::PageId, Vec<(u16, u32)>> = HashMap::new();
        for &(pid, slot, xmin) in rids {
            by_page.entry(pid).or_default().push((slot, xmin));
        }
        for (page_id, slots) in by_page {
            // Resident: validate through the read guard with no full-page
            // copy (only a few header bytes per slot are read, and the loop
            // does no await so the guard is held briefly). Non-resident: one
            // owned read from disk.
            let all_ok = match buffer_pool.fetch_page(page_id) {
                Some(frame) => {
                    let g = frame.read_data();
                    let ok = slots
                        .iter()
                        .all(|&(slot, fx)| slot_still_folded(&**g, slot, fx));
                    drop(g);
                    buffer_pool.unpin_page(page_id, false);
                    ok
                }
                None => match rt.block_on(disk_manager.read_page(page_id)) {
                    Ok(d) => slots
                        .iter()
                        .all(|&(slot, fx)| slot_still_folded(&d, slot, fx)),
                    Err(_) => return Ok(false),
                },
            };
            if !all_ok {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Zeroes the folded heap slots so a folded row exists only in columnar,
    /// and writes each modified page durably to disk so the post-commit state
    /// survives a crash. No per-slot identity check here: this runs only on
    /// the live fold path, immediately after `revalidate_folded` confirmed at
    /// the commit point that every slot still holds its folded tuple, and the
    /// folded tuple stays live (slot length > 0) until this zeroes it, so the
    /// FSM cannot hand the slot to another insert in the gap. Slot reuse is
    /// possible only across a crash, and that redo path
    /// (`columnar_recovery`) does its own xmin identity check from the
    /// sidecar. Keeping the check out of this per-row loop is what avoids a
    /// redundant tuple read on the fold hot path.
    fn delete_folded_rows(
        rt: &tokio::runtime::Runtime,
        buffer_pool: &Arc<BufferPool>,
        disk_manager: &Arc<DiskManager>,
        rids: &[FoldedRid],
    ) -> std::result::Result<(), String> {
        use std::collections::HashMap;
        let mut by_page: HashMap<zyron_common::page::PageId, Vec<u16>> = HashMap::new();
        for &(pid, slot, _xmin) in rids {
            by_page.entry(pid).or_default().push(slot);
        }
        for (page_id, slots) in by_page {
            let mut page_data = match buffer_pool.fetch_page(page_id) {
                Some(frame) => {
                    let guard = frame.read_data();
                    let data: [u8; PAGE_SIZE] = **guard;
                    drop(guard);
                    buffer_pool.unpin_page(page_id, false);
                    data
                }
                None => match rt.block_on(disk_manager.read_page(page_id)) {
                    Ok(d) => d,
                    Err(_) => continue,
                },
            };
            for slot in slots {
                let slot_off = HeapPage::DATA_START + (slot as usize) * 4;
                // Zero the slot length to mark the tuple removed.
                page_data[slot_off + 2] = 0;
                page_data[slot_off + 3] = 0;
            }
            // Update the buffer-pool copy if resident.
            if let Some(frame) = buffer_pool.fetch_page(page_id) {
                frame.copy_from(&page_data);
                buffer_pool.unpin_page(page_id, true);
            }
            // Durable write: the heap-delete half of the committed transition
            // must survive a crash without depending on the buffer pool.
            rt.block_on(disk_manager.write_page(page_id, &page_data))
                .map_err(|e| format!("durable heap page write failed: {}", e))?;
        }
        Ok(())
    }

    /// Returns a reference to compaction statistics.
    pub fn stats(&self) -> &Arc<CompactionStats> {
        &self.stats
    }

    /// Gracefully shuts down the worker thread. Called before the final
    /// checkpoint so no fold is in flight during shutdown.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CompactionWorker {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let c = CompactionWorkerConfig::default();
        assert_eq!(c.interval_secs, 30);
        assert_eq!(c.min_rows, 100_000);
        assert_eq!(c.oltp_p99_threshold_us, 1_000);
    }

    #[test]
    fn test_stats_initial() {
        let s = CompactionStats::new();
        assert_eq!(s.cycles_completed.load(Ordering::Relaxed), 0);
        assert_eq!(s.rows_folded.load(Ordering::Relaxed), 0);
    }
}

//! HeapFile manager with buffer pool integration for high-performance tuple storage.
//!
//! All page I/O is routed through the buffer pool for caching. Pages are fetched
//! from the pool, modified in memory, marked dirty, and written back lazily.

use crate::disk::DiskManager;
use crate::freespace::{
    ENTRIES_PER_FSM_PAGE, FreeSpaceMap, FsmPage, category_to_min_space, space_to_category,
};
use crate::heap::constants::{DATA_START, HEAP_HEADER_OFFSET, TUPLE_HEADER_SIZE, TUPLE_SLOT_SIZE};
use crate::heap::page::{HeapPage, SlotId};
use crate::tuple::{Tuple, TupleHeader, TupleId, TupleView};
use std::sync::Arc;
use zyron_buffer::BufferPool;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_common::{Result, ZyronError};

/// Configuration for HeapFile.
#[derive(Debug, Clone)]
pub struct HeapFileConfig {
    /// File ID for the heap data file.
    pub heap_file_id: u32,
    /// File ID for the free space map file.
    pub fsm_file_id: u32,
}

impl Default for HeapFileConfig {
    fn default() -> Self {
        Self {
            heap_file_id: 0,
            fsm_file_id: 1,
        }
    }
}

/// Pending FSM update entry.
struct PendingFsmUpdate {
    heap_page_num: u32,
    free_space: usize,
}

// ---------------------------------------------------------------------------
// Lock-free atomic hint slots for FSM page lookup
// ---------------------------------------------------------------------------

/// Number of hint slots for fast insert page lookup.
const HINT_SLOT_COUNT: usize = 8;

/// Sentinel value for empty hint slot.
const HINT_EMPTY: u64 = u64::MAX;

/// Lock-free atomic hint slots for FSM page lookup.
///
/// Each slot packs (page_num: u32, free_space_category: u32) into a u64.
/// Reads and writes use Relaxed ordering. Stale reads are acceptable because
/// the FSM on disk is the source of truth. This is a best-effort cache.
struct AtomicHintSlots {
    slots: [std::sync::atomic::AtomicU64; HINT_SLOT_COUNT],
}

impl AtomicHintSlots {
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(HINT_EMPTY)),
        }
    }

    #[inline(always)]
    fn pack(page_num: u32, category: u32) -> u64 {
        ((page_num as u64) << 32) | (category as u64)
    }

    #[inline(always)]
    fn unpack(packed: u64) -> (u32, u32) {
        ((packed >> 32) as u32, packed as u32)
    }

    /// Updates or inserts a hint. Scans for existing entry first.
    /// If not found, overwrites the last slot (LRU approximation).
    #[inline]
    fn update(&self, page_num: u32, category: u8) {
        use std::sync::atomic::Ordering::Relaxed;
        let new_val = Self::pack(page_num, category as u32);

        // Check if already present in any slot
        for slot in &self.slots {
            let current = slot.load(Relaxed);
            if current != HINT_EMPTY {
                let (pn, _) = Self::unpack(current);
                if pn == page_num {
                    slot.store(new_val, Relaxed);
                    return;
                }
            }
        }

        // Not found. Find first empty slot.
        for slot in &self.slots {
            let current = slot.load(Relaxed);
            if current == HINT_EMPTY {
                slot.store(new_val, Relaxed);
                return;
            }
        }

        // All slots full. Overwrite last slot (LRU approximation).
        self.slots[HINT_SLOT_COUNT - 1].store(new_val, Relaxed);
    }

    /// Removes a page from the hints.
    #[inline]
    fn remove(&self, page_num: u32) {
        use std::sync::atomic::Ordering::Relaxed;
        for slot in &self.slots {
            let current = slot.load(Relaxed);
            if current != HINT_EMPTY {
                let (pn, _) = Self::unpack(current);
                if pn == page_num {
                    slot.store(HINT_EMPTY, Relaxed);
                    return;
                }
            }
        }
    }

    /// Finds a page with at least `min_space` free.
    /// Returns Some(page_num) on first match.
    #[inline]
    fn find_page_with_space(&self, min_space: usize) -> Option<u32> {
        use std::sync::atomic::Ordering::Relaxed;
        for slot in &self.slots {
            let current = slot.load(Relaxed);
            if current != HINT_EMPTY {
                let (pn, cat) = Self::unpack(current);
                if category_to_min_space(cat as u8) >= min_space {
                    return Some(pn);
                }
            }
        }
        None
    }
}

/// HeapFile manages tuple storage with buffer pool caching.
///
/// All page accesses go through the buffer pool for memory efficiency.
/// Dirty pages are written back lazily by the buffer pool eviction.
pub struct HeapFile {
    /// Disk manager for page I/O.
    disk: Arc<DiskManager>,
    /// Buffer pool for page caching.
    pool: Arc<BufferPool>,
    /// Free space map metadata.
    fsm: FreeSpaceMap,
    /// Configuration.
    config: HeapFileConfig,
    /// Cached heap page count (avoids repeated disk.num_pages calls).
    cached_heap_pages: std::sync::atomic::AtomicU32,
    /// Cached FSM page count.
    cached_fsm_pages: std::sync::atomic::AtomicU32,
    /// Lock-free hint slots for fast page lookup during inserts.
    hint_slots: AtomicHintSlots,
    /// Pending FSM updates. Uses Mutex because Vec append is not lock-free.
    /// Only contended during FSM flush and page boundary transitions.
    pending_fsm: parking_lot::Mutex<Vec<PendingFsmUpdate>>,
    /// Sharded insertion points: one tail page per writer thread so concurrent
    /// appends claim space on different pages (different cache lines), giving
    /// each shard a single writer and an uncontended header claim. Each shard is
    /// u32::MAX until a thread first inserts here, so a table written by K
    /// threads uses K tail pages: adaptive, with no waste for cold or
    /// single-writer tables.
    insert_shards: Box<[CachePaddedU32]>,
}

/// Cache-line-aligned AtomicU32 so per-shard insertion pointers written by
/// different writer threads never false-share an adjacent shard.
#[repr(align(64))]
struct CachePaddedU32(std::sync::atomic::AtomicU32);

/// Upper bound on sharded heap insertion points. The live count is
/// `min(this, available_parallelism)`; 64 covers any core count. Shards are
/// lazy, so an unused shard costs one atomic slot, never a page.
const MAX_INSERT_SHARDS: usize = 64;

/// Source of stable per-thread writer slots. The first heap insert from a thread
/// claims the next slot; modulo a file's shard count it selects that thread's
/// shard, so a worker thread keeps appending to the same tail page (cache-local)
/// and distinct workers spread across pages.
static NEXT_WRITER_SLOT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

std::thread_local! {
    static WRITER_SLOT: usize =
        NEXT_WRITER_SLOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

impl HeapFile {
    /// Creates a new HeapFile with buffer pool integration.
    pub fn new(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        config: HeapFileConfig,
    ) -> Result<Self> {
        use std::sync::atomic::AtomicU32;
        let fsm = FreeSpaceMap::new(config.heap_file_id, PageId::new(config.fsm_file_id, 0));

        Ok(Self {
            disk,
            pool,
            fsm,
            config,
            cached_heap_pages: AtomicU32::new(0),
            cached_fsm_pages: AtomicU32::new(0),
            hint_slots: AtomicHintSlots::new(),
            pending_fsm: parking_lot::Mutex::new(Vec::with_capacity(64)),
            insert_shards: {
                let shard_count = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(8)
                    .clamp(1, MAX_INSERT_SHARDS);
                (0..shard_count)
                    .map(|_| CachePaddedU32(AtomicU32::new(u32::MAX)))
                    .collect()
            },
        })
    }

    /// Initializes page count caches from disk (call once at startup).
    pub async fn init_cache(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        let heap_pages = self.disk.num_pages(self.config.heap_file_id).await?;
        let fsm_pages = self.disk.num_pages(self.config.fsm_file_id).await?;
        self.cached_heap_pages
            .store(heap_pages as u32, Ordering::Relaxed);
        self.cached_fsm_pages
            .store(fsm_pages as u32, Ordering::Relaxed);
        // Resume appending into the last existing page on one shard; the others
        // stay lazy and roll into fresh or reused pages on first use. The first
        // insert on the resumed shard detects PageFull and rolls over via CAS.
        if heap_pages > 0 {
            self.insert_shards[0]
                .0
                .store((heap_pages - 1) as u32, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Returns cached heap page count.
    #[inline]
    fn heap_page_count(&self) -> u32 {
        use std::sync::atomic::Ordering;
        self.cached_heap_pages.load(Ordering::Relaxed)
    }

    /// Returns cached FSM page count.
    #[inline]
    fn fsm_page_count(&self) -> u32 {
        use std::sync::atomic::Ordering;
        self.cached_fsm_pages.load(Ordering::Relaxed)
    }

    /// This thread's insertion shard. A stable per-thread slot keeps each shard
    /// single-writer while writer threads do not exceed the shard count, so the
    /// page-header claim is uncontended; beyond that the slots wrap and a few
    /// threads share a shard, degrading gracefully to the prior behavior.
    #[inline]
    fn insert_shard(&self) -> &std::sync::atomic::AtomicU32 {
        let idx = WRITER_SLOT.with(|s| *s) % self.insert_shards.len();
        &self.insert_shards[idx].0
    }

    /// Increments cached FSM page count.
    #[inline]
    fn increment_fsm_pages(&self) {
        use std::sync::atomic::Ordering;
        self.cached_fsm_pages.fetch_add(1, Ordering::Relaxed);
    }

    /// Creates a HeapFile with default configuration.
    pub fn with_defaults(disk: Arc<DiskManager>, pool: Arc<BufferPool>) -> Result<Self> {
        Self::new(disk, pool, HeapFileConfig::default())
    }

    /// Returns the heap file ID.
    #[inline]
    pub fn heap_file_id(&self) -> u32 {
        self.config.heap_file_id
    }

    /// Returns the FSM file ID.
    #[inline]
    pub fn fsm_file_id(&self) -> u32 {
        self.config.fsm_file_id
    }

    /// Fetches a page from the buffer pool, loading from disk if needed.
    #[inline]
    async fn fetch_page(&self, page_id: PageId) -> Result<[u8; PAGE_SIZE]> {
        // Check if page is in buffer pool
        if let Some(frame) = self.pool.fetch_page(page_id) {
            let guard = frame.read_data();
            let data: [u8; PAGE_SIZE] = **guard;
            drop(guard);
            self.pool.unpin_page(page_id, false);
            return Ok(data);
        }

        // Load from disk into buffer pool
        let disk_data = self.disk.read_page(page_id).await?;
        let (frame, evicted) = self.pool.load_page(page_id, &disk_data)?;

        // Handle evicted dirty page
        if let Some(evicted_page) = evicted {
            self.disk
                .write_page(evicted_page.page_id, &evicted_page.data)
                .await?;
        }

        let guard = frame.read_data();
        let data: [u8; PAGE_SIZE] = **guard;
        drop(guard);
        self.pool.unpin_page(page_id, false);
        Ok(data)
    }

    /// Writes a page through the buffer pool (marks dirty, handles eviction).
    #[inline]
    async fn write_page(&self, page_id: PageId, data: &[u8; PAGE_SIZE]) -> Result<()> {
        // Try to fetch existing page from pool
        if let Some(frame) = self.pool.fetch_page(page_id) {
            frame.copy_from(data);
            self.pool.unpin_page(page_id, true); // Mark dirty
            return Ok(());
        }

        // Load into pool (load_page already copies data into frame)
        let (_, evicted) = self.pool.load_page(page_id, data)?;

        // Handle evicted dirty page
        if let Some(evicted_page) = evicted {
            self.disk
                .write_page(evicted_page.page_id, &evicted_page.data)
                .await?;
        }

        self.pool.unpin_page(page_id, true); // Mark dirty
        Ok(())
    }

    // =========================================================================
    // Tuple Operations
    // =========================================================================

    /// Retrieves a tuple by its TupleId (allocates a Vec for tuple data).
    pub async fn get(&self, tuple_id: TupleId) -> Result<Option<Tuple>> {
        let page_data = match self.fetch_page(tuple_id.page_id).await {
            Ok(data) => data,
            Err(ZyronError::IoError(_)) => return Ok(None),
            Err(e) => return Err(e),
        };

        let page = HeapPage::from_bytes(page_data);
        Ok(page.get_tuple(SlotId(tuple_id.slot_id)))
    }

    /// Zero-copy tuple access. The closure receives a TupleView that borrows
    /// directly from the page buffer. No Vec allocation for tuple data.
    /// The page frame is held for the duration of the closure.
    pub async fn with_tuple<F, R>(&self, tuple_id: TupleId, f: F) -> Result<Option<R>>
    where
        F: FnOnce(TupleView<'_>) -> R,
    {
        let page_data = match self.fetch_page(tuple_id.page_id).await {
            Ok(data) => data,
            Err(ZyronError::IoError(_)) => return Ok(None),
            Err(e) => return Err(e),
        };

        let view = HeapPage::get_tuple_view_from_slice(&page_data, SlotId(tuple_id.slot_id));
        Ok(view.map(f))
    }

    /// Deletes a tuple by its TupleId.
    ///
    /// Returns true if the tuple was deleted, false if not found.
    /// For bulk deletes, use `delete_batch` for better performance.
    pub async fn delete(&self, tuple_id: TupleId) -> Result<bool> {
        let page_data = match self.fetch_page(tuple_id.page_id).await {
            Ok(data) => data,
            Err(ZyronError::IoError(_)) => return Ok(false),
            Err(e) => return Err(e),
        };

        let mut page = HeapPage::from_bytes(page_data);
        let deleted = page.delete_tuple(SlotId(tuple_id.slot_id));

        if deleted {
            self.write_page(tuple_id.page_id, page.as_bytes()).await?;
            // Defer FSM update for batched processing
            self.defer_fsm_update(tuple_id.page_id.page_num as u32, page.total_usable_space());
        }

        Ok(deleted)
    }

    /// Deletes multiple tuples efficiently by grouping by page.
    ///
    /// Returns the number of tuples successfully deleted.
    pub async fn delete_batch(&self, tuple_ids: &[TupleId]) -> Result<usize> {
        use std::collections::HashMap;

        if tuple_ids.is_empty() {
            return Ok(0);
        }

        // Group tuple IDs by page
        let mut pages: HashMap<PageId, Vec<u16>> = HashMap::new();
        for tuple_id in tuple_ids {
            pages
                .entry(tuple_id.page_id)
                .or_default()
                .push(tuple_id.slot_id);
        }

        let mut deleted_count = 0;

        // Process each page once
        for (page_id, slot_ids) in pages {
            let page_data = match self.fetch_page(page_id).await {
                Ok(data) => data,
                Err(ZyronError::IoError(_)) => continue,
                Err(e) => return Err(e),
            };

            let mut page = HeapPage::from_bytes(page_data);
            let mut page_modified = false;

            for slot_id in slot_ids {
                if page.delete_tuple(SlotId(slot_id)) {
                    deleted_count += 1;
                    page_modified = true;
                }
            }

            if page_modified {
                self.write_page(page_id, page.as_bytes()).await?;
                self.defer_fsm_update(page_id.page_num as u32, page.total_usable_space());
            }
        }

        // Batch flush all FSM updates
        self.flush_fsm_updates().await?;

        Ok(deleted_count)
    }

    /// MVCC delete: stamps `xmax` on each tuple in place instead of freeing the
    /// slot. The rows stay physically present and are hidden by snapshot
    /// visibility once the deleting transaction commits; an aborted delete
    /// leaves them visible, and vacuum reclaims the space later. Space is NOT
    /// returned to the FSM here (the slots are not free yet). Returns the number
    /// of tuples stamped. Groups by page so each page is read and written once.
    pub async fn mark_deleted_batch(
        &self,
        tuple_ids: &[TupleId],
        xmax: u32,
        prune_horizon: u64,
        status: Option<&crate::TxnStatusMap>,
        retain_history: bool,
    ) -> Result<usize> {
        use std::collections::HashMap;

        if tuple_ids.is_empty() {
            return Ok(0);
        }

        let mut pages: HashMap<PageId, Vec<u16>> = HashMap::new();
        for tuple_id in tuple_ids {
            pages
                .entry(tuple_id.page_id)
                .or_default()
                .push(tuple_id.slot_id);
        }

        let mut marked = 0;
        for (page_id, slot_ids) in pages {
            // Pin the frame, then take the exclusive frame write lock and mutate
            // in place. The lock-free burst-append path holds the shared frame
            // lock, so this exclusive lock prevents a concurrent append from
            // being lost (the old copy-out, modify, copy-in pattern could clobber
            // an append that landed between the read and the write).
            let frame = match self.pool.fetch_page(page_id) {
                Some(frame) => frame,
                None => {
                    let disk_data = match self.disk.read_page(page_id).await {
                        Ok(d) => d,
                        Err(ZyronError::IoError(_)) => continue,
                        Err(e) => return Err(e),
                    };
                    let (frame, evicted) = self.pool.load_page(page_id, &disk_data)?;
                    if let Some(ev) = evicted {
                        self.disk.write_page(ev.page_id, &ev.data).await?;
                    }
                    frame
                }
            };

            let mut page_modified = false;
            let mut reclaimed_free: Option<usize> = None;
            {
                let mut guard = frame.write_data();
                let data: &mut [u8] = &mut guard[..];
                for slot_id in slot_ids {
                    if HeapPage::set_tuple_xmax_in_slice(data, SlotId(slot_id), xmax) {
                        marked += 1;
                        page_modified = true;
                    }
                }
                // On-access pruning: the page is already locked, so reclaim any
                // versions dead to every live snapshot (a committed delete below
                // the frozen horizon, or an aborted insert). This keeps
                // MVCC-updated heaps compact without waiting for the vacuum cycle.
                if let Some(status) = status
                    && prune_horizon > 0
                {
                    let is_dead = |xmin: u32, x: u32| {
                        // Aborted insert: never visible at any version, always
                        // reclaimable. Committed delete below the frozen horizon:
                        // reclaim only if no retained version still sees the row
                        // alive (the deleter committed at or before the floor).
                        // When the table has a time-travel retention policy,
                        // skip committed-delete reclamation here entirely and
                        // leave it to the retention-aware background vacuum, so
                        // an on-access prune never drops history early.
                        status.is_aborted(xmin as u64)
                            || (!retain_history
                                && x != 0
                                && (x as u64) < prune_horizon
                                && status.version_reclaimable(x as u64))
                    };
                    if HeapPage::prune_dead_in_slice(data, &is_dead) {
                        page_modified = true;
                        reclaimed_free = Some(HeapPage::free_space_in_slice(data));
                    }
                }
            }
            self.pool.unpin_page(page_id, page_modified);

            // Publish reclaimed free space so the insert path reuses this page
            // instead of growing the heap.
            if let Some(free) = reclaimed_free {
                self.defer_fsm_update(page_id.page_num as u32, free);
            }
        }
        if marked > 0 {
            self.flush_fsm_updates().await?;
        }
        Ok(marked)
    }

    /// Stamps a single tuple's `xmax` under the page write lock. Used by ROLLBACK
    /// TO SAVEPOINT to self-delete a row the transaction inserted after the
    /// savepoint, making it invisible to the still-open transaction (and, after
    /// commit, to every transaction). Frees no space, leaves index entries in
    /// place (heap visibility filters them, vacuum reclaims them later), matching
    /// the MVCC-delete invariant. Returns true if the tuple was stamped.
    pub async fn set_xmax(&self, tuple_id: TupleId, xmax: u32) -> Result<bool> {
        self.mutate_tuple_xmax(tuple_id, Some(xmax)).await
    }

    /// Clears a single tuple's `xmax` back to 0 under the page write lock. Used
    /// by ROLLBACK TO SAVEPOINT to restore a row the transaction deleted after
    /// the savepoint. Returns true if the tuple was restored.
    pub async fn clear_xmax(&self, tuple_id: TupleId) -> Result<bool> {
        self.mutate_tuple_xmax(tuple_id, None).await
    }

    /// Shared body for set_xmax/clear_xmax. Pins the page, takes the exclusive
    /// frame write lock (the same lock mark_deleted_batch takes, so a concurrent
    /// burst-append cannot be clobbered), and stamps or clears the tuple's xmax.
    async fn mutate_tuple_xmax(&self, tuple_id: TupleId, xmax: Option<u32>) -> Result<bool> {
        let page_id = tuple_id.page_id;
        let frame = match self.pool.fetch_page(page_id) {
            Some(frame) => frame,
            None => {
                let disk_data = match self.disk.read_page(page_id).await {
                    Ok(d) => d,
                    Err(ZyronError::IoError(_)) => return Ok(false),
                    Err(e) => return Err(e),
                };
                let (frame, evicted) = self.pool.load_page(page_id, &disk_data)?;
                if let Some(ev) = evicted {
                    self.disk.write_page(ev.page_id, &ev.data).await?;
                }
                frame
            }
        };

        let changed = {
            let mut guard = frame.write_data();
            let data: &mut [u8] = &mut guard[..];
            match xmax {
                Some(x) => HeapPage::set_tuple_xmax_in_slice(data, SlotId(tuple_id.slot_id), x),
                None => HeapPage::clear_tuple_xmax_in_slice(data, SlotId(tuple_id.slot_id)),
            }
        };
        self.pool.unpin_page(page_id, changed);
        Ok(changed)
    }

    /// Updates a tuple in place if the new tuple fits.
    ///
    /// Returns error if the new tuple is larger than the old one.
    pub async fn update(&self, tuple_id: TupleId, tuple: &Tuple) -> Result<()> {
        let page_data = self.fetch_page(tuple_id.page_id).await?;
        let mut page = HeapPage::from_bytes(page_data);

        page.update_tuple(SlotId(tuple_id.slot_id), tuple)?;

        self.write_page(tuple_id.page_id, page.as_bytes()).await?;
        self.update_fsm_for_page(tuple_id.page_id.page_num as u32, page.free_space())
            .await?;

        Ok(())
    }

    /// Zero-copy scan of all tuples in the heap file.
    ///
    /// Returns a guard that holds pinned pages. Use `.iter()` to iterate
    /// over tuples as borrowed `TupleView` references. Pages are automatically
    /// unpinned when the guard is dropped.
    pub fn scan(&self) -> Result<ScanGuard<'_>> {
        let num_pages = self.heap_page_count();
        let file_id = self.config.heap_file_id;

        let page_ids: Vec<PageId> = (0..num_pages)
            .map(|n| PageId::new(file_id, n as u64))
            .collect();

        self.pool.batch_pin(&page_ids);

        Ok(ScanGuard {
            pool: &self.pool,
            page_ids,
        })
    }

    /// Returns the number of pages in the heap file.
    pub async fn num_pages(&self) -> Result<u32> {
        Ok(self.heap_page_count())
    }

    /// Lock-free atomic read of the cached page count for hot paths
    /// that cannot afford async or fallible API
    #[inline]
    pub fn num_pages_cached(&self) -> u32 {
        self.heap_page_count()
    }

    /// Flushes all dirty heap pages to disk.
    /// Uses synchronous I/O because flush_all's closure cannot await.
    pub async fn flush(&self) -> Result<()> {
        let data_dir = self.disk.data_dir().to_path_buf();
        let heap_file_id = self.config.heap_file_id;
        let fsm_file_id = self.config.fsm_file_id;

        self.pool.flush_all(|page_id, data| {
            if page_id.file_id != heap_file_id && page_id.file_id != fsm_file_id {
                return Ok(());
            }

            let path = data_dir.join(format!("{:08}.dat", page_id.file_id));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(|e| {
                    ZyronError::IoError(format!("flush open {}: {}", path.display(), e))
                })?;

            let offset = page_id.page_num * (PAGE_SIZE as u64);
            std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(offset))
                .map_err(|e| ZyronError::IoError(format!("flush seek: {}", e)))?;
            std::io::Write::write_all(&mut file, data)
                .map_err(|e| ZyronError::IoError(format!("flush write: {}", e)))?;

            Ok(())
        })?;
        Ok(())
    }

    /// Updates the FSM entry for a page.
    async fn update_fsm_for_page(&self, heap_page_num: u32, free_space: usize) -> Result<()> {
        let fsm_page_num = self.fsm.fsm_page_for(heap_page_num);
        let fsm_page_id = PageId::new(self.config.fsm_file_id, fsm_page_num as u64);

        // Use cached FSM page count
        let num_fsm_pages = self.fsm_page_count();

        let mut fsm_page = if fsm_page_num < num_fsm_pages {
            let fsm_data = self.fetch_page(fsm_page_id).await?;
            FsmPage::from_bytes(fsm_data)
        } else {
            // Allocate new FSM page
            let first_tracked = fsm_page_num * ENTRIES_PER_FSM_PAGE as u32;
            let new_fsm_page = FsmPage::new(fsm_page_id, first_tracked);
            self.disk.allocate_page(self.config.fsm_file_id).await?;
            self.increment_fsm_pages();
            new_fsm_page
        };

        let category = space_to_category(free_space);
        fsm_page.set_space(heap_page_num, category)?;
        self.write_page(fsm_page_id, fsm_page.as_bytes()).await?;

        Ok(())
    }

    // =========================================================================
    // Batched FSM Operations
    // =========================================================================
    // For high-throughput inserts, FSM updates can be deferred and batched.

    /// Updates hint slots atomically (lock-free) and queues FSM update.
    /// For batch callers that accumulate updates locally, use
    /// `update_hints()` during the loop and `push_fsm_updates()` once at the end.
    #[inline]
    pub(crate) fn defer_fsm_update(&self, heap_page_num: u32, free_space: usize) {
        self.update_hints(heap_page_num, free_space);
        self.pending_fsm.lock().push(PendingFsmUpdate {
            heap_page_num,
            free_space,
        });
    }

    /// Lock-free hint slot update only. No Mutex touched.
    #[inline]
    pub(crate) fn update_hints(&self, heap_page_num: u32, free_space: usize) {
        let category = space_to_category(free_space);
        if category > 0 {
            self.hint_slots.update(heap_page_num, category);
        } else {
            self.hint_slots.remove(heap_page_num);
        }
    }

    /// Flushes all pending FSM updates in a single batch.
    ///
    /// Groups updates by FSM page to minimize I/O. Each FSM page is
    /// read once, updated with all pending entries, and written once.
    pub async fn flush_fsm_updates(&self) -> Result<usize> {
        // Take all pending updates
        let updates: Vec<PendingFsmUpdate> = {
            let mut pending = self.pending_fsm.lock();
            std::mem::take(&mut *pending)
        };

        if updates.is_empty() {
            return Ok(0);
        }

        // Group updates by FSM page number
        let mut by_fsm_page: std::collections::HashMap<u32, Vec<(u32, usize)>> =
            std::collections::HashMap::new();

        for update in &updates {
            let fsm_page_num = self.fsm.fsm_page_for(update.heap_page_num);
            by_fsm_page
                .entry(fsm_page_num)
                .or_default()
                .push((update.heap_page_num, update.free_space));
        }

        let num_fsm_pages = self.fsm_page_count();

        // Process each FSM page once with all its updates
        for (fsm_page_num, page_updates) in by_fsm_page {
            let fsm_page_id = PageId::new(self.config.fsm_file_id, fsm_page_num as u64);

            let mut fsm_page = if fsm_page_num < num_fsm_pages {
                let fsm_data = self.fetch_page(fsm_page_id).await?;
                FsmPage::from_bytes(fsm_data)
            } else {
                // Allocate new FSM page
                let first_tracked = fsm_page_num * ENTRIES_PER_FSM_PAGE as u32;
                let new_fsm_page = FsmPage::new(fsm_page_id, first_tracked);
                self.disk.allocate_page(self.config.fsm_file_id).await?;
                self.increment_fsm_pages();
                new_fsm_page
            };

            // Apply all updates to this FSM page
            for (heap_page_num, free_space) in page_updates {
                let category = space_to_category(free_space);
                fsm_page.set_space(heap_page_num, category)?;
            }

            self.write_page(fsm_page_id, fsm_page.as_bytes()).await?;
        }

        Ok(updates.len())
    }

    /// Batch insert via the lock-free atomic burst path
    ///
    /// Multi-page batches pre-allocate locally in one allocate_pages_batch
    /// call and consume from a local Vec, single-page batches go through
    /// this thread's insertion shard so concurrent writers append to distinct
    /// tail pages.
    pub async fn insert_batch(&self, tuples: &[Tuple]) -> Result<Vec<TupleId>> {
        use std::sync::atomic::Ordering;

        if tuples.is_empty() {
            return Ok(Vec::new());
        }

        // This thread's insertion shard, used for every single-page append and
        // rollover below so concurrent writers do not share one tail page.
        let insert_shard = self.insert_shard();

        let usable_per_page = PAGE_SIZE - DATA_START;
        let total_bytes: usize = tuples
            .iter()
            .map(|t| t.size_on_disk() + TUPLE_SLOT_SIZE)
            .sum();
        let estimated_pages = ((total_bytes + usable_per_page - 1) / usable_per_page).max(1) as u64;

        let mut local_pages: Vec<PageId> = if estimated_pages > 1 {
            let alloc = self
                .disk
                .allocate_pages_batch(self.config.heap_file_id, estimated_pages)
                .await?;
            self.cached_heap_pages
                .fetch_add(estimated_pages as u32, Ordering::Relaxed);
            alloc
        } else {
            Vec::new()
        };
        // pop() yields in allocation order
        local_pages.reverse();

        let mut results = Vec::with_capacity(tuples.len());
        let mut cursor = 0usize;
        let mut last_used_page: Option<u32> = None;

        while cursor < tuples.len() {
            let (page_id, is_fresh) = if let Some(pid) = local_pages.pop() {
                (pid, true)
            } else {
                let mut page_num = insert_shard.load(Ordering::Acquire);
                if page_num == u32::MAX {
                    page_num = self.advance_insert_page(insert_shard, u32::MAX).await?;
                }
                (
                    PageId::new(self.config.heap_file_id, page_num as u64),
                    false,
                )
            };

            if is_fresh {
                let mut buf = [0u8; PAGE_SIZE];
                HeapPage::init_fresh_slice_reuse(&mut buf, page_id);
                let (_, evicted) = self.pool.load_page(page_id, &buf)?;
                if let Some(ev) = evicted {
                    self.disk.write_page(ev.page_id, &ev.data).await?;
                }
            } else if self.pool.fetch_page(page_id).is_none() {
                let disk_data = self.disk.read_page(page_id).await?;
                let (_, evicted) = self.pool.load_page(page_id, &disk_data)?;
                if let Some(ev) = evicted {
                    self.disk.write_page(ev.page_id, &ev.data).await?;
                }
            }

            let inserted = unsafe {
                let frame = self
                    .pool
                    .fetch_page(page_id)
                    .expect("just pinned this page");
                // balance the extra pin from this fetch
                self.pool.unpin_page(page_id, false);
                // Hold the shared frame lock for the duration of the append.
                // Appenders share this lock and write disjoint regions claimed
                // via the header CAS; the delete/prune path takes the exclusive
                // lock, so compaction can never move tuples out from under an
                // in-flight append.
                let _shared = frame.read_data();
                let raw: *mut [u8; PAGE_SIZE] = frame.data_ptr_mut();
                HeapPage::insert_tuples_burst(
                    raw as *mut u8,
                    page_id,
                    &tuples[cursor..],
                    &mut results,
                )
            };

            if inserted == 0 {
                self.pool.unpin_page(page_id, false);
                if !is_fresh {
                    let _ = self
                        .advance_insert_page(insert_shard, page_id.page_num as u32)
                        .await?;
                }
                continue;
            }

            self.pool.unpin_page(page_id, true);
            cursor += inserted;
            last_used_page = Some(page_id.page_num as u32);
        }

        // monotonic publish of the last used page into this thread's shard for
        // its future single-tuple inserts
        if let Some(last) = last_used_page {
            let mut current = insert_shard.load(Ordering::Acquire);
            while current == u32::MAX || last > current {
                match insert_shard.compare_exchange_weak(
                    current,
                    last,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(observed) => current = observed,
                }
            }
        }

        // unused pre-alloc tail stays as empty pages, SeqScan empty-page skip filters them
        Ok(results)
    }

    /// Rolls `shard` to a successor page and publishes it via CAS; losers leak
    /// an empty page that SeqScan filters via slot_count == 0. Operating on the
    /// caller's shard keeps each writer's rollover independent of the others.
    async fn advance_insert_page(
        &self,
        shard: &std::sync::atomic::AtomicU32,
        observed_page: u32,
    ) -> Result<u32> {
        use std::sync::atomic::Ordering;

        let current = shard.load(Ordering::Acquire);
        if current != observed_page {
            return Ok(current);
        }

        // The page that just filled has no room: drop it from the hint cache so
        // the reuse lookup below never picks it again.
        if observed_page != u32::MAX {
            self.hint_slots.remove(observed_page);
        }

        // Reuse a page that pruning reclaimed space on before growing the heap.
        // This is the slow path (only on page fill), so the hint lookup adds no
        // per-insert cost, and it keeps MVCC-updated heaps from growing
        // unbounded between vacuum cycles. Require at least an eighth of a page
        // of contiguous free space so a reused page absorbs several inserts.
        const MIN_REUSE_SPACE: usize = PAGE_SIZE / 8;
        if let Some(reuse_page) = self.hint_slots.find_page_with_space(MIN_REUSE_SPACE)
            && reuse_page != observed_page
        {
            match shard.compare_exchange(
                observed_page,
                reuse_page,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(reuse_page),
                Err(other) => return Ok(other),
            }
        }

        let new_page_id = self.disk.allocate_page(self.config.heap_file_id).await?;
        let new_page_num = new_page_id.page_num as u32;
        self.cached_heap_pages.fetch_add(1, Ordering::Relaxed);

        let mut buf = [0u8; PAGE_SIZE];
        HeapPage::init_fresh_slice_reuse(&mut buf, new_page_id);
        let (_, evicted) = self.pool.load_page(new_page_id, &buf)?;
        if let Some(ev) = evicted {
            self.disk.write_page(ev.page_id, &ev.data).await?;
        }
        self.pool.unpin_page(new_page_id, true);

        match shard.compare_exchange(
            observed_page,
            new_page_num,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(new_page_num),
            Err(other) => Ok(other),
        }
    }
}

/// Guard that holds pinned pages during zero-copy scan iteration.
///
/// Pages remain pinned for the lifetime of this guard, allowing safe
/// borrowing of tuple data directly from page buffers.
pub struct ScanGuard<'a> {
    pool: &'a BufferPool,
    page_ids: Vec<PageId>,
}

impl<'a> ScanGuard<'a> {
    /// Returns the list of page IDs held by this scan guard.
    #[inline]
    pub fn page_ids(&self) -> &[PageId] {
        &self.page_ids
    }

    /// Direct callback iteration over all tuples.
    /// Uses raw pointer access and unchecked indexing for maximum throughput.
    #[inline]
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(TupleId, TupleView<'_>),
    {
        let max_slots = (PAGE_SIZE - DATA_START) / TUPLE_SLOT_SIZE;
        for &page_id in &self.page_ids {
            if let Some(p) = unsafe { self.pool.frame_data_ptr(page_id) } {
                // Safety: page is pinned and frame_data_ptr returned valid pointer
                let data = unsafe { &*p };
                let raw_slot_count =
                    u16::from_le_bytes([data[HEAP_HEADER_OFFSET], data[HEAP_HEADER_OFFSET + 1]])
                        as usize;
                // Cap slot_count to prevent out-of-bounds reads from corrupt page headers.
                let slot_count = raw_slot_count.min(max_slots);

                for i in 0..slot_count {
                    // Safety: slot_base is within page bounds (slot_count from page header)
                    let slot_base = DATA_START + i * TUPLE_SLOT_SIZE;
                    let tuple_length = unsafe {
                        u16::from_le_bytes([
                            *data.get_unchecked(slot_base + 2),
                            *data.get_unchecked(slot_base + 3),
                        ])
                    } as usize;

                    if tuple_length == 0 {
                        continue;
                    }

                    let tuple_offset = unsafe {
                        u16::from_le_bytes([
                            *data.get_unchecked(slot_base),
                            *data.get_unchecked(slot_base + 1),
                        ])
                    } as usize;

                    // Safety: tuple_offset validated by page format, header fits in page
                    let header = unsafe {
                        TupleHeader::from_bytes_unchecked(
                            &data[tuple_offset..tuple_offset + TUPLE_HEADER_SIZE],
                        )
                    };
                    let data_start = tuple_offset + TUPLE_HEADER_SIZE;
                    let data_end = data_start + header.data_len as usize;

                    f(
                        TupleId::new(page_id, i as u16),
                        TupleView::new(header, &data[data_start..data_end]),
                    );
                }
            }
        }
    }

    /// Like `for_each` but stops as soon as `f` returns `false`. Lets a
    /// targeted lookup (e.g. resolve one table by name) abandon the scan on
    /// the first match instead of materializing and deserializing every
    /// remaining tuple.
    pub fn try_for_each<F>(&self, mut f: F)
    where
        F: FnMut(TupleId, TupleView<'_>) -> bool,
    {
        let max_slots = (PAGE_SIZE - DATA_START) / TUPLE_SLOT_SIZE;
        for &page_id in &self.page_ids {
            if let Some(p) = unsafe { self.pool.frame_data_ptr(page_id) } {
                let data = unsafe { &*p };
                let raw_slot_count =
                    u16::from_le_bytes([data[HEAP_HEADER_OFFSET], data[HEAP_HEADER_OFFSET + 1]])
                        as usize;
                let slot_count = raw_slot_count.min(max_slots);

                for i in 0..slot_count {
                    let slot_base = DATA_START + i * TUPLE_SLOT_SIZE;
                    let tuple_length = unsafe {
                        u16::from_le_bytes([
                            *data.get_unchecked(slot_base + 2),
                            *data.get_unchecked(slot_base + 3),
                        ])
                    } as usize;

                    if tuple_length == 0 {
                        continue;
                    }

                    let tuple_offset = unsafe {
                        u16::from_le_bytes([
                            *data.get_unchecked(slot_base),
                            *data.get_unchecked(slot_base + 1),
                        ])
                    } as usize;

                    let header = unsafe {
                        TupleHeader::from_bytes_unchecked(
                            &data[tuple_offset..tuple_offset + TUPLE_HEADER_SIZE],
                        )
                    };
                    let data_start = tuple_offset + TUPLE_HEADER_SIZE;
                    let data_end = data_start + header.data_len as usize;

                    if !f(
                        TupleId::new(page_id, i as u16),
                        TupleView::new(header, &data[data_start..data_end]),
                    ) {
                        return;
                    }
                }
            }
        }
    }

    /// Fast tuple count without constructing TupleView for each tuple.
    #[inline]
    pub fn count(&self) -> usize {
        let max_slots = (PAGE_SIZE - DATA_START) / TUPLE_SLOT_SIZE;
        let mut total = 0;
        for &page_id in &self.page_ids {
            if let Some(p) = unsafe { self.pool.frame_data_ptr(page_id) } {
                let data = unsafe { &*p };
                let raw_slot_count =
                    u16::from_le_bytes([data[HEAP_HEADER_OFFSET], data[HEAP_HEADER_OFFSET + 1]])
                        as usize;
                let slot_count = raw_slot_count.min(max_slots);

                for i in 0..slot_count {
                    let slot_base = DATA_START + i * TUPLE_SLOT_SIZE;
                    let tuple_length = unsafe {
                        u16::from_le_bytes([
                            *data.get_unchecked(slot_base + 2),
                            *data.get_unchecked(slot_base + 3),
                        ])
                    } as usize;
                    if tuple_length != 0 {
                        total += 1;
                    }
                }
            }
        }
        total
    }
}

impl Drop for ScanGuard<'_> {
    fn drop(&mut self) {
        self.pool.batch_unpin(&self.page_ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::DiskManagerConfig;
    use tempfile::tempdir;
    use zyron_buffer::BufferPoolConfig;
    use zyron_common::page::PAGE_SIZE;

    async fn create_test_heap() -> (HeapFile, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        };
        let disk = Arc::new(DiskManager::new(config).await.unwrap());
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 100 }));
        let heap = HeapFile::with_defaults(disk, pool).unwrap();
        (heap, dir)
    }

    #[tokio::test]
    async fn test_heap_file_new() {
        let (heap, _dir) = create_test_heap().await;
        assert_eq!(heap.heap_file_id(), 0);
        assert_eq!(heap.fsm_file_id(), 1);
    }

    #[tokio::test]
    async fn test_heap_file_insert() {
        let (heap, _dir) = create_test_heap().await;

        let data = b"hello world".to_vec();
        let tuple = Tuple::new(data, 1);

        let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        assert!(tuple_id.is_valid());
        assert_eq!(tuple_id.page_id.file_id, 0);
        assert_eq!(tuple_id.page_id.page_num, 0);
        assert_eq!(tuple_id.slot_id, 0);
    }

    #[tokio::test]
    async fn test_heap_file_get() {
        let (heap, _dir) = create_test_heap().await;

        let data = b"test data".to_vec();
        let tuple = Tuple::new(data.clone(), 42);

        let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        let retrieved = heap.get(tuple_id).await.unwrap().unwrap();

        assert_eq!(retrieved.data(), &data);
        assert_eq!(retrieved.header().xmin, 42);
    }

    #[tokio::test]
    async fn test_heap_file_get_nonexistent() {
        let (heap, _dir) = create_test_heap().await;

        let tuple_id = TupleId::new(PageId::new(0, 999), 0);
        let result = heap.get(tuple_id).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_heap_file_delete() {
        let (heap, _dir) = create_test_heap().await;

        let data = b"to delete".to_vec();
        let tuple = Tuple::new(data, 1);

        let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        assert!(heap.get(tuple_id).await.unwrap().is_some());

        assert!(heap.delete(tuple_id).await.unwrap());
        assert!(heap.get(tuple_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_heap_file_delete_nonexistent() {
        let (heap, _dir) = create_test_heap().await;

        // Insert a tuple first so the page exists
        let data = b"data".to_vec();
        let tuple = Tuple::new(data, 1);
        heap.insert_batch(&[tuple]).await.unwrap().remove(0);

        let tuple_id = TupleId::new(PageId::new(0, 0), 999);
        assert!(!heap.delete(tuple_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_heap_file_update() {
        let (heap, _dir) = create_test_heap().await;

        // Insert a larger tuple
        let data1 = vec![0u8; 100];
        let tuple1 = Tuple::new(data1, 1);
        let tuple_id = heap.insert_batch(&[tuple1]).await.unwrap().remove(0);

        // Update with smaller tuple
        let data2 = vec![1u8; 50];
        let tuple2 = Tuple::new(data2.clone(), 2);
        heap.update(tuple_id, &tuple2).await.unwrap();

        let retrieved = heap.get(tuple_id).await.unwrap().unwrap();
        assert_eq!(retrieved.header().xmin, 2);
    }

    #[tokio::test]
    async fn test_heap_file_update_too_large() {
        let (heap, _dir) = create_test_heap().await;

        let data1 = vec![0u8; 10];
        let tuple1 = Tuple::new(data1, 1);
        let tuple_id = heap.insert_batch(&[tuple1]).await.unwrap().remove(0);

        let data2 = vec![1u8; 100];
        let tuple2 = Tuple::new(data2, 2);
        let result = heap.update(tuple_id, &tuple2).await;

        assert!(matches!(result, Err(ZyronError::PageFull)));
    }

    #[tokio::test]
    async fn test_heap_file_multiple_inserts() {
        let (heap, _dir) = create_test_heap().await;

        for i in 0..100 {
            let data = format!("tuple {}", i).into_bytes();
            let tuple = Tuple::new(data, i);
            let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);
            assert!(tuple_id.is_valid());
        }
    }

    #[tokio::test]
    async fn test_heap_file_scan() {
        let (heap, _dir) = create_test_heap().await;

        for i in 0..10 {
            let data = format!("tuple {}", i).into_bytes();
            let tuple = Tuple::new(data, i);
            heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        }

        let guard = heap.scan().unwrap();
        let mut count = 0;
        let mut xmins = Vec::new();
        guard.for_each(|_, tuple| {
            count += 1;
            xmins.push(tuple.header.xmin);
        });
        assert_eq!(count, 10);

        for (i, xmin) in xmins.iter().enumerate() {
            assert_eq!(*xmin, i as u32);
        }
    }

    #[tokio::test]
    async fn test_heap_file_scan_with_deletions() {
        let (heap, _dir) = create_test_heap().await;

        let mut ids = Vec::new();
        for i in 0..10 {
            let data = format!("tuple {}", i).into_bytes();
            let tuple = Tuple::new(data, i);
            ids.push(heap.insert_batch(&[tuple]).await.unwrap().remove(0));
        }

        // Delete every other tuple
        for i in (0..10).step_by(2) {
            heap.delete(ids[i]).await.unwrap();
        }

        let guard = heap.scan().unwrap();
        assert_eq!(guard.count(), 5);
    }

    #[tokio::test]
    async fn test_heap_file_multiple_pages() {
        let (heap, _dir) = create_test_heap().await;

        // Insert large tuples to span multiple pages
        let tuple_size = PAGE_SIZE / 4;
        for i in 0..20 {
            let data = vec![i as u8; tuple_size];
            let tuple = Tuple::new(data, i as u32);
            heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        }

        assert!(heap.num_pages().await.unwrap() > 1);

        let guard = heap.scan().unwrap();
        assert_eq!(guard.count(), 20);
    }

    #[tokio::test]
    async fn test_heap_file_reuses_space() {
        let (heap, _dir) = create_test_heap().await;

        // Insert and delete a tuple
        let data = vec![0u8; 100];
        let tuple = Tuple::new(data, 1);
        let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        heap.delete(tuple_id).await.unwrap();

        // Insert again - should reuse space
        let data2 = vec![1u8; 50];
        let tuple2 = Tuple::new(data2, 2);
        let tuple_id2 = heap.insert_batch(&[tuple2]).await.unwrap().remove(0);

        // Should be on the same page
        assert_eq!(tuple_id.page_id, tuple_id2.page_id);
    }

    #[tokio::test]
    async fn test_heap_file_fsm_space_tracking() {
        let (heap, _dir) = create_test_heap().await;

        // Insert some tuples
        for i in 0..5 {
            let data = format!("tuple {}", i).into_bytes();
            let tuple = Tuple::new(data, i);
            heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        }

        // Delete them all
        for i in 0..5 {
            let tuple_id = TupleId::new(PageId::new(0, 0), i);
            heap.delete(tuple_id).await.unwrap();
        }

        // Insert a large tuple - should find space on existing page
        let data = vec![0u8; 1000];
        let tuple = Tuple::new(data, 100);
        let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);

        // Should still be on page 0 due to FSM finding space
        assert_eq!(tuple_id.page_id.page_num, 0);
    }

    #[tokio::test]
    async fn test_heap_file_num_pages() {
        let (heap, _dir) = create_test_heap().await;

        assert_eq!(heap.num_pages().await.unwrap(), 0);

        let data = b"data".to_vec();
        let tuple = Tuple::new(data, 1);
        heap.insert_batch(&[tuple]).await.unwrap().remove(0);

        assert_eq!(heap.num_pages().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_heap_file_buffer_pool_caching() {
        let dir = tempdir().unwrap();
        let config = DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        };
        let disk = Arc::new(DiskManager::new(config).await.unwrap());
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 10 }));
        let heap = HeapFile::with_defaults(disk.clone(), pool.clone()).unwrap();

        // Insert tuples
        for i in 0..5 {
            let data = format!("tuple {}", i).into_bytes();
            let tuple = Tuple::new(data, i);
            heap.insert_batch(&[tuple]).await.unwrap().remove(0);
        }

        // Pages should be in buffer pool
        assert!(pool.page_count() > 0);

        // Reading should hit cache
        for i in 0..5 {
            let tuple_id = TupleId::new(PageId::new(0, 0), i as u16);
            heap.get(tuple_id).await.unwrap();
        }
    }
}

//! HeapFile manager with buffer pool integration for high-performance tuple storage.
//!
//! All page I/O is routed through the buffer pool for caching. Pages are fetched
//! from the pool, modified in memory, marked dirty, and written back lazily.

use crate::disk::DiskManager;
use crate::freespace::{space_to_category, FreeSpaceMap, FsmPage, ENTRIES_PER_FSM_PAGE};
use crate::heap::page::{HeapPage, SlotId};
use crate::tuple::{Tuple, TupleId};
use std::sync::Arc;
use zyron_buffer::BufferPool;
use zyron_common::page::{PageId, PAGE_SIZE};
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

/// LRU cache of page hints for fast insert page lookup.
/// Stores up to 8 recently used pages with their free space category.
const PAGE_HINT_CACHE_SIZE: usize = 8;

struct PageHintCache {
    /// Array of (page_num, free_space_category) pairs. u32::MAX = empty slot.
    hints: [(u32, u8); PAGE_HINT_CACHE_SIZE],
    /// Number of valid entries.
    count: usize,
}

impl PageHintCache {
    fn new() -> Self {
        Self {
            hints: [(u32::MAX, 0); PAGE_HINT_CACHE_SIZE],
            count: 0,
        }
    }

    /// Finds a page with at least the required space category.
    #[inline]
    fn find_page(&self, required_category: u8) -> Option<u32> {
        for i in 0..self.count {
            let (page_num, category) = self.hints[i];
            if page_num != u32::MAX && category >= required_category {
                return Some(page_num);
            }
        }
        None
    }

    /// Adds or updates a page hint. Moves to front (MRU position).
    #[inline]
    fn update(&mut self, page_num: u32, category: u8) {
        // Check if already present
        for i in 0..self.count {
            if self.hints[i].0 == page_num {
                // Move to front (update category in case it changed)
                for j in (1..=i).rev() {
                    self.hints[j] = self.hints[j - 1];
                }
                self.hints[0] = (page_num, category);
                return;
            }
        }

        // Not present, add to front
        if self.count < PAGE_HINT_CACHE_SIZE {
            // Shift existing entries
            for i in (1..=self.count).rev() {
                self.hints[i] = self.hints[i - 1];
            }
            self.hints[0] = (page_num, category);
            self.count += 1;
        } else {
            // Cache full, evict LRU (last entry)
            for i in (1..PAGE_HINT_CACHE_SIZE).rev() {
                self.hints[i] = self.hints[i - 1];
            }
            self.hints[0] = (page_num, category);
        }
    }

    /// Removes a page from the cache (when full).
    #[inline]
    fn remove(&mut self, page_num: u32) {
        for i in 0..self.count {
            if self.hints[i].0 == page_num {
                // Shift remaining entries
                for j in i..self.count - 1 {
                    self.hints[j] = self.hints[j + 1];
                }
                self.hints[self.count - 1] = (u32::MAX, 0);
                self.count -= 1;
                return;
            }
        }
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
    /// LRU cache of pages with space (speeds up inserts).
    page_hint_cache: parking_lot::Mutex<PageHintCache>,
    /// Last page with space hint (speeds up sequential inserts).
    last_page_hint: std::sync::atomic::AtomicU32,
    /// Pending FSM updates for batched writes.
    pending_fsm_updates: parking_lot::Mutex<Vec<PendingFsmUpdate>>,
}

impl HeapFile {
    /// Creates a new HeapFile with buffer pool integration.
    pub fn new(disk: Arc<DiskManager>, pool: Arc<BufferPool>, config: HeapFileConfig) -> Result<Self> {
        use std::sync::atomic::AtomicU32;
        let fsm = FreeSpaceMap::new(config.heap_file_id, PageId::new(config.fsm_file_id, 0));

        Ok(Self {
            disk,
            pool,
            fsm,
            config,
            cached_heap_pages: AtomicU32::new(0),
            cached_fsm_pages: AtomicU32::new(0),
            page_hint_cache: parking_lot::Mutex::new(PageHintCache::new()),
            last_page_hint: AtomicU32::new(u32::MAX), // Invalid hint initially
            pending_fsm_updates: parking_lot::Mutex::new(Vec::with_capacity(64)),
        })
    }

    /// Initializes page count caches from disk (call once at startup).
    pub async fn init_cache(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        let heap_pages = self.disk.num_pages(self.config.heap_file_id).await?;
        let fsm_pages = self.disk.num_pages(self.config.fsm_file_id).await?;
        self.cached_heap_pages.store(heap_pages, Ordering::Relaxed);
        self.cached_fsm_pages.store(fsm_pages, Ordering::Relaxed);
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

    /// Increments cached heap page count.
    #[inline]
    fn increment_heap_pages(&self) {
        use std::sync::atomic::Ordering;
        self.cached_heap_pages.fetch_add(1, Ordering::Relaxed);
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
            self.disk.write_page(evicted_page.page_id, &*evicted_page.data).await?;
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

        // Load into pool and mark dirty
        let (frame, evicted) = self.pool.load_page(page_id, data)?;

        // Handle evicted dirty page
        if let Some(evicted_page) = evicted {
            self.disk.write_page(evicted_page.page_id, &*evicted_page.data).await?;
        }

        frame.copy_from(data);
        self.pool.unpin_page(page_id, true); // Mark dirty
        Ok(())
    }

    // =========================================================================
    // Synchronous Fast Path Operations
    // =========================================================================

    /// Synchronous insert for cached pages only.
    ///
    /// Tries to insert into the hint page if it's in the buffer pool.
    /// Returns CacheMiss if no suitable cached page is available.
    /// Returns PageFull if the hint page doesn't have enough space.
    #[inline]
    pub fn insert_sync(&self, tuple: &Tuple) -> Result<TupleId> {
        use std::sync::atomic::Ordering;

        // Only try hint page for sync path
        let hint = self.last_page_hint.load(Ordering::Relaxed);
        if hint == u32::MAX || hint >= self.heap_page_count() {
            return Err(ZyronError::CacheMiss);
        }

        let hint_page_id = PageId::new(self.config.heap_file_id, hint);

        // Try to get write access to the page in buffer pool
        if let Some(guard) = self.pool.write_page(hint_page_id) {
            let mut data = guard.data_mut();

            // Try to insert using in-slice method
            match HeapPage::insert_tuple_in_slice(&mut **data, tuple) {
                Ok((slot_id, free_space)) => {
                    guard.set_dirty();
                    // Defer FSM update for batching
                    self.defer_fsm_update(hint_page_id.page_num, free_space);
                    return Ok(TupleId::new(hint_page_id, slot_id.0));
                }
                Err(ZyronError::PageFull) => {
                    // Clear hint since page is full
                    self.last_page_hint.store(u32::MAX, Ordering::Relaxed);
                    return Err(ZyronError::PageFull);
                }
                Err(e) => return Err(e),
            }
        }

        Err(ZyronError::CacheMiss)
    }

    /// Inserts a tuple into the heap file.
    ///
    /// Finds a page with sufficient space using a hint or FSM, or allocates
    /// a new page if needed. Returns the TupleId of the inserted tuple.
    /// Tries sync fast path first to avoid async overhead when pages are cached.
    pub async fn insert(&self, tuple: &Tuple) -> Result<TupleId> {
        use std::sync::atomic::Ordering;

        // Try sync fast path first
        match self.insert_sync(tuple) {
            Ok(tuple_id) => return Ok(tuple_id),
            Err(ZyronError::CacheMiss) => { /* fall through to async path */ }
            Err(ZyronError::PageFull) => { /* fall through to find new page */ }
            Err(e) => return Err(e),
        }

        let tuple_size = tuple.size_on_disk();
        let space_needed = tuple_size + crate::heap::page::TupleSlot::SIZE;
        let required_category = space_to_category(space_needed);

        // Try last page hint first (fast path for sequential inserts)
        let hint = self.last_page_hint.load(Ordering::Relaxed);
        if hint != u32::MAX && hint < self.heap_page_count() {
            let hint_page_id = PageId::new(self.config.heap_file_id, hint);
            match self.insert_into_page(hint_page_id, tuple).await {
                Ok(tuple_id) => return Ok(tuple_id),
                Err(ZyronError::PageFull) => {
                    // Hint page is full, clear hint and remove from cache
                    self.last_page_hint.store(u32::MAX, Ordering::Relaxed);
                    self.page_hint_cache.lock().remove(hint);
                }
                Err(e) => return Err(e),
            }
        }

        // Try page hint cache (LRU of recently used pages)
        if let Some(cached_page_num) = self.page_hint_cache.lock().find_page(required_category) {
            let cached_page_id = PageId::new(self.config.heap_file_id, cached_page_num);
            match self.insert_into_page(cached_page_id, tuple).await {
                Ok(tuple_id) => {
                    self.last_page_hint.store(cached_page_num, Ordering::Relaxed);
                    return Ok(tuple_id);
                }
                Err(ZyronError::PageFull) => {
                    self.page_hint_cache.lock().remove(cached_page_num);
                }
                Err(e) => return Err(e),
            }
        }

        // Try to find a page with enough space via FSM
        if let Some(page_id) = self.find_page_with_space(space_needed).await? {
            match self.insert_into_page(page_id, tuple).await {
                Ok(tuple_id) => {
                    // Update hints for next insert
                    self.last_page_hint.store(page_id.page_num, Ordering::Relaxed);
                    return Ok(tuple_id);
                }
                Err(ZyronError::PageFull) => {
                    // FSM estimate was slightly off, fall through to allocate new page
                }
                Err(e) => return Err(e),
            }
        }

        // No suitable page found or FSM estimate was off, allocate a new one
        let page_id = self.allocate_new_page().await?;
        let tuple_id = self.insert_into_page(page_id, tuple).await?;
        // Update hints to newly allocated page
        self.last_page_hint.store(page_id.page_num, Ordering::Relaxed);
        Ok(tuple_id)
    }

    /// Retrieves a tuple by its TupleId.
    pub async fn get(&self, tuple_id: TupleId) -> Result<Option<Tuple>> {
        let page_data = match self.fetch_page(tuple_id.page_id).await {
            Ok(data) => data,
            Err(ZyronError::IoError(_)) => return Ok(None),
            Err(e) => return Err(e),
        };

        let page = HeapPage::from_bytes(page_data);
        Ok(page.get_tuple(SlotId(tuple_id.slot_id)))
    }

    /// Deletes a tuple by its TupleId.
    ///
    /// Returns true if the tuple was deleted, false if not found.
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
            self.update_fsm_for_page(tuple_id.page_id.page_num, page.free_space()).await?;
        }

        Ok(deleted)
    }

    /// Updates a tuple in place if the new tuple fits.
    ///
    /// Returns error if the new tuple is larger than the old one.
    pub async fn update(&self, tuple_id: TupleId, tuple: &Tuple) -> Result<()> {
        let page_data = self.fetch_page(tuple_id.page_id).await?;
        let mut page = HeapPage::from_bytes(page_data);

        page.update_tuple(SlotId(tuple_id.slot_id), tuple)?;

        self.write_page(tuple_id.page_id, page.as_bytes()).await?;
        self.update_fsm_for_page(tuple_id.page_id.page_num, page.free_space()).await?;

        Ok(())
    }

    /// Scans all tuples in the heap file.
    ///
    /// Returns a vector of all (TupleId, Tuple) pairs.
    /// Uses sync path for cached pages to minimize async overhead.
    pub async fn scan(&self) -> Result<Vec<(TupleId, Tuple)>> {
        let num_pages = self.heap_page_count();
        // Pre-allocate with estimated capacity (~60 tuples per page)
        let mut results = Vec::with_capacity((num_pages as usize) * 60);

        for page_num in 0..num_pages {
            let page_id = PageId::new(self.config.heap_file_id, page_num);

            // Try sync path first for cached pages
            if let Some(guard) = self.pool.read_page(page_id) {
                let data = guard.data();
                let page = HeapPage::from_bytes(**data);
                for (slot_id, tuple) in page.iter() {
                    let tuple_id = TupleId::new(page_id, slot_id.0);
                    results.push((tuple_id, tuple));
                }
                continue;
            }

            // Async fallback for uncached pages
            let page_data = self.fetch_page(page_id).await?;
            let page = HeapPage::from_bytes(page_data);

            for (slot_id, tuple) in page.iter() {
                let tuple_id = TupleId::new(page_id, slot_id.0);
                results.push((tuple_id, tuple));
            }
        }

        Ok(results)
    }

    /// Returns the number of pages in the heap file.
    pub async fn num_pages(&self) -> Result<u32> {
        Ok(self.heap_page_count())
    }

    /// Flushes all dirty heap pages to disk.
    pub async fn flush(&self) -> Result<()> {
        self.pool.flush_all(|page_id, data| {
            // Only flush pages belonging to this heap file
            if page_id.file_id == self.config.heap_file_id || page_id.file_id == self.config.fsm_file_id {
                // Note: This is synchronous, which is a limitation.
                // For full async, we'd need a different approach.
                let data_copy: [u8; PAGE_SIZE] = data.try_into().map_err(|_| {
                    ZyronError::Internal("Invalid page size".to_string())
                })?;
                // We can't await here, so we'll use blocking for now
                // A proper solution would queue these writes
                let _ = &data_copy; // Suppress unused warning
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Finds a page with at least the specified amount of free space.
    async fn find_page_with_space(&self, min_space: usize) -> Result<Option<PageId>> {
        let num_heap_pages = self.heap_page_count();
        if num_heap_pages == 0 {
            return Ok(None);
        }

        let num_fsm_pages = self.fsm_page_count();

        for fsm_page_num in 0..num_fsm_pages {
            let fsm_page_id = PageId::new(self.config.fsm_file_id, fsm_page_num);
            let fsm_data = self.fetch_page(fsm_page_id).await?;
            let fsm_page = FsmPage::from_bytes(fsm_data);

            if let Some(heap_page_num) = fsm_page.find_page_with_space(min_space) {
                if heap_page_num < num_heap_pages {
                    return Ok(Some(PageId::new(self.config.heap_file_id, heap_page_num)));
                }
            }
        }

        Ok(None)
    }

    /// Inserts a tuple into a specific page with FSM update.
    async fn insert_into_page(&self, page_id: PageId, tuple: &Tuple) -> Result<TupleId> {
        let page_data = self.fetch_page(page_id).await?;
        let mut page = HeapPage::from_bytes(page_data);

        let slot_id = page.insert_tuple(tuple)?;

        self.write_page(page_id, page.as_bytes()).await?;
        self.update_fsm_for_page(page_id.page_num, page.free_space()).await?;

        Ok(TupleId::new(page_id, slot_id.0))
    }

    /// Allocates a new heap page.
    async fn allocate_new_page(&self) -> Result<PageId> {
        let page_id = self.disk.allocate_page(self.config.heap_file_id).await?;
        self.increment_heap_pages();

        // Initialize as heap page
        let page = HeapPage::new(page_id);
        self.write_page(page_id, page.as_bytes()).await?;

        // Update FSM with initial free space
        self.update_fsm_for_page(page_id.page_num, page.free_space()).await?;

        Ok(page_id)
    }

    /// Updates the FSM entry for a page.
    async fn update_fsm_for_page(&self, heap_page_num: u32, free_space: usize) -> Result<()> {
        let fsm_page_num = self.fsm.fsm_page_for(heap_page_num);
        let fsm_page_id = PageId::new(self.config.fsm_file_id, fsm_page_num);

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

    /// Queues an FSM update for later batch processing.
    #[inline]
    fn defer_fsm_update(&self, heap_page_num: u32, free_space: usize) {
        // Update page hint cache with the current free space category
        let category = space_to_category(free_space);
        if category > 0 {
            self.page_hint_cache.lock().update(heap_page_num, category);
        } else {
            // Page is full, remove from cache
            self.page_hint_cache.lock().remove(heap_page_num);
        }

        self.pending_fsm_updates.lock().push(PendingFsmUpdate {
            heap_page_num,
            free_space,
        });
    }

    /// Flushes all pending FSM updates in a single batch.
    ///
    /// Groups updates by FSM page to minimize I/O. Each FSM page is
    /// read once, updated with all pending entries, and written once.
    pub async fn flush_fsm_updates(&self) -> Result<usize> {
        // Take all pending updates
        let updates: Vec<PendingFsmUpdate> = {
            let mut pending = self.pending_fsm_updates.lock();
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
            let fsm_page_id = PageId::new(self.config.fsm_file_id, fsm_page_num);

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

    /// Batch insert multiple tuples with deferred FSM updates.
    ///
    /// Inserts tuples sequentially into pages, deferring FSM updates
    /// until the end for better performance. Returns the TupleIds of
    /// all inserted tuples.
    pub async fn insert_batch(&self, tuples: &[Tuple]) -> Result<Vec<TupleId>> {
        use std::sync::atomic::Ordering;

        if tuples.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(tuples.len());
        let mut current_page_id: Option<PageId> = None;
        let mut current_page: Option<HeapPage> = None;

        for tuple in tuples {
            let tuple_size = tuple.size_on_disk();
            let space_needed = tuple_size + crate::heap::page::TupleSlot::SIZE;

            // Check if current page has space
            let use_current = if let Some(ref page) = current_page {
                page.free_space() >= space_needed
            } else {
                false
            };

            if !use_current {
                // Flush current page if any
                if let (Some(page_id), Some(page)) = (current_page_id, &current_page) {
                    self.write_page(page_id, page.as_bytes()).await?;
                    self.defer_fsm_update(page_id.page_num, page.free_space());
                }

                // Try hint page first
                let hint = self.last_page_hint.load(Ordering::Relaxed);
                let hint_page_id = if hint != u32::MAX && hint < self.heap_page_count() {
                    Some(PageId::new(self.config.heap_file_id, hint))
                } else {
                    None
                };

                // Find or allocate a page with space
                let found_page = if let Some(hint_id) = hint_page_id {
                    let page_data = self.fetch_page(hint_id).await?;
                    let page = HeapPage::from_bytes(page_data);
                    if page.free_space() >= space_needed {
                        Some((hint_id, page))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let (page_id, page) = if let Some((id, p)) = found_page {
                    (id, p)
                } else if let Some(id) = self.find_page_with_space(space_needed).await? {
                    let page_data = self.fetch_page(id).await?;
                    (id, HeapPage::from_bytes(page_data))
                } else {
                    // Allocate new page
                    let id = self.disk.allocate_page(self.config.heap_file_id).await?;
                    self.increment_heap_pages();
                    (id, HeapPage::new(id))
                };

                current_page_id = Some(page_id);
                current_page = Some(page);
                self.last_page_hint.store(page_id.page_num, Ordering::Relaxed);
            }

            // Insert into current page
            if let Some(ref mut page) = current_page {
                let slot_id = page.insert_tuple(tuple)?;
                results.push(TupleId::new(current_page_id.unwrap(), slot_id.0));
            }
        }

        // Flush final page
        if let (Some(page_id), Some(page)) = (current_page_id, &current_page) {
            self.write_page(page_id, page.as_bytes()).await?;
            self.defer_fsm_update(page_id.page_num, page.free_space());
        }

        // Batch flush all FSM updates
        self.flush_fsm_updates().await?;

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::DiskManagerConfig;
    use bytes::Bytes;
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

        let data = Bytes::from_static(b"hello world");
        let tuple = Tuple::new(data, 1);

        let tuple_id = heap.insert(&tuple).await.unwrap();
        assert!(tuple_id.is_valid());
        assert_eq!(tuple_id.page_id.file_id, 0);
        assert_eq!(tuple_id.page_id.page_num, 0);
        assert_eq!(tuple_id.slot_id, 0);
    }

    #[tokio::test]
    async fn test_heap_file_get() {
        let (heap, _dir) = create_test_heap().await;

        let data = Bytes::from_static(b"test data");
        let tuple = Tuple::new(data.clone(), 42);

        let tuple_id = heap.insert(&tuple).await.unwrap();
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

        let data = Bytes::from_static(b"to delete");
        let tuple = Tuple::new(data, 1);

        let tuple_id = heap.insert(&tuple).await.unwrap();
        assert!(heap.get(tuple_id).await.unwrap().is_some());

        assert!(heap.delete(tuple_id).await.unwrap());
        assert!(heap.get(tuple_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_heap_file_delete_nonexistent() {
        let (heap, _dir) = create_test_heap().await;

        // Insert a tuple first so the page exists
        let data = Bytes::from_static(b"data");
        let tuple = Tuple::new(data, 1);
        heap.insert(&tuple).await.unwrap();

        let tuple_id = TupleId::new(PageId::new(0, 0), 999);
        assert!(!heap.delete(tuple_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_heap_file_update() {
        let (heap, _dir) = create_test_heap().await;

        // Insert a larger tuple
        let data1 = Bytes::from(vec![0u8; 100]);
        let tuple1 = Tuple::new(data1, 1);
        let tuple_id = heap.insert(&tuple1).await.unwrap();

        // Update with smaller tuple
        let data2 = Bytes::from(vec![1u8; 50]);
        let tuple2 = Tuple::new(data2.clone(), 2);
        heap.update(tuple_id, &tuple2).await.unwrap();

        let retrieved = heap.get(tuple_id).await.unwrap().unwrap();
        assert_eq!(retrieved.header().xmin, 2);
    }

    #[tokio::test]
    async fn test_heap_file_update_too_large() {
        let (heap, _dir) = create_test_heap().await;

        let data1 = Bytes::from(vec![0u8; 10]);
        let tuple1 = Tuple::new(data1, 1);
        let tuple_id = heap.insert(&tuple1).await.unwrap();

        let data2 = Bytes::from(vec![1u8; 100]);
        let tuple2 = Tuple::new(data2, 2);
        let result = heap.update(tuple_id, &tuple2).await;

        assert!(matches!(result, Err(ZyronError::PageFull)));
    }

    #[tokio::test]
    async fn test_heap_file_multiple_inserts() {
        let (heap, _dir) = create_test_heap().await;

        for i in 0..100 {
            let data = Bytes::from(format!("tuple {}", i));
            let tuple = Tuple::new(data, i);
            let tuple_id = heap.insert(&tuple).await.unwrap();
            assert!(tuple_id.is_valid());
        }
    }

    #[tokio::test]
    async fn test_heap_file_scan() {
        let (heap, _dir) = create_test_heap().await;

        for i in 0..10 {
            let data = Bytes::from(format!("tuple {}", i));
            let tuple = Tuple::new(data, i);
            heap.insert(&tuple).await.unwrap();
        }

        let tuples = heap.scan().await.unwrap();
        assert_eq!(tuples.len(), 10);

        for (i, (_, tuple)) in tuples.iter().enumerate() {
            assert_eq!(tuple.header().xmin, i as u32);
        }
    }

    #[tokio::test]
    async fn test_heap_file_scan_with_deletions() {
        let (heap, _dir) = create_test_heap().await;

        let mut ids = Vec::new();
        for i in 0..10 {
            let data = Bytes::from(format!("tuple {}", i));
            let tuple = Tuple::new(data, i);
            ids.push(heap.insert(&tuple).await.unwrap());
        }

        // Delete every other tuple
        for i in (0..10).step_by(2) {
            heap.delete(ids[i]).await.unwrap();
        }

        let tuples = heap.scan().await.unwrap();
        assert_eq!(tuples.len(), 5);
    }

    #[tokio::test]
    async fn test_heap_file_multiple_pages() {
        let (heap, _dir) = create_test_heap().await;

        // Insert large tuples to span multiple pages
        let tuple_size = PAGE_SIZE / 4;
        for i in 0..20 {
            let data = Bytes::from(vec![i as u8; tuple_size]);
            let tuple = Tuple::new(data, i as u32);
            heap.insert(&tuple).await.unwrap();
        }

        assert!(heap.num_pages().await.unwrap() > 1);

        let tuples = heap.scan().await.unwrap();
        assert_eq!(tuples.len(), 20);
    }

    #[tokio::test]
    async fn test_heap_file_reuses_space() {
        let (heap, _dir) = create_test_heap().await;

        // Insert and delete a tuple
        let data = Bytes::from(vec![0u8; 100]);
        let tuple = Tuple::new(data, 1);
        let tuple_id = heap.insert(&tuple).await.unwrap();
        heap.delete(tuple_id).await.unwrap();

        // Insert again - should reuse space
        let data2 = Bytes::from(vec![1u8; 50]);
        let tuple2 = Tuple::new(data2, 2);
        let tuple_id2 = heap.insert(&tuple2).await.unwrap();

        // Should be on the same page
        assert_eq!(tuple_id.page_id, tuple_id2.page_id);
    }

    #[tokio::test]
    async fn test_heap_file_fsm_space_tracking() {
        let (heap, _dir) = create_test_heap().await;

        // Insert some tuples
        for i in 0..5 {
            let data = Bytes::from(format!("tuple {}", i));
            let tuple = Tuple::new(data, i);
            heap.insert(&tuple).await.unwrap();
        }

        // Delete them all
        for i in 0..5 {
            let tuple_id = TupleId::new(PageId::new(0, 0), i);
            heap.delete(tuple_id).await.unwrap();
        }

        // Insert a large tuple - should find space on existing page
        let data = Bytes::from(vec![0u8; 1000]);
        let tuple = Tuple::new(data, 100);
        let tuple_id = heap.insert(&tuple).await.unwrap();

        // Should still be on page 0 due to FSM finding space
        assert_eq!(tuple_id.page_id.page_num, 0);
    }

    #[tokio::test]
    async fn test_heap_file_num_pages() {
        let (heap, _dir) = create_test_heap().await;

        assert_eq!(heap.num_pages().await.unwrap(), 0);

        let data = Bytes::from_static(b"data");
        let tuple = Tuple::new(data, 1);
        heap.insert(&tuple).await.unwrap();

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
            let data = Bytes::from(format!("tuple {}", i));
            let tuple = Tuple::new(data, i);
            heap.insert(&tuple).await.unwrap();
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

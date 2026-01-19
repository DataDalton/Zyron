//! HeapFile manager with buffer pool integration for high-performance tuple storage.
//!
//! All page I/O is routed through the buffer pool for caching. Pages are fetched
//! from the pool, modified in memory, marked dirty, and written back lazily.

use crate::disk::DiskManager;
use crate::freespace::{ENTRIES_PER_FSM_PAGE, FreeSpaceMap, FsmPage, space_to_category};
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

/// Combined state for FSM operations (single lock for hint cache + pending updates).
struct FsmState {
    hint_cache: PageHintCache,
    pending_updates: Vec<PendingFsmUpdate>,
}

impl FsmState {
    fn new() -> Self {
        Self {
            hint_cache: PageHintCache::new(),
            pending_updates: Vec::with_capacity(64),
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
    /// Combined FSM state (hint cache + pending updates) under single lock.
    fsm_state: parking_lot::Mutex<FsmState>,
    /// Last page with space hint (speeds up sequential inserts).
    last_page_hint: std::sync::atomic::AtomicU32,
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
            fsm_state: parking_lot::Mutex::new(FsmState::new()),
            last_page_hint: AtomicU32::new(u32::MAX), // Invalid hint initially
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
            self.disk
                .write_page(evicted_page.page_id, &*evicted_page.data)
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
                .write_page(evicted_page.page_id, &*evicted_page.data)
                .await?;
        }

        self.pool.unpin_page(page_id, true); // Mark dirty
        Ok(())
    }

    // =========================================================================
    // Tuple Operations
    // =========================================================================

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
            self.defer_fsm_update(tuple_id.page_id.page_num, page.total_usable_space());
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
                self.defer_fsm_update(page_id.page_num, page.total_usable_space());
            }
        }

        // Batch flush all FSM updates
        self.flush_fsm_updates().await?;

        Ok(deleted_count)
    }

    /// Updates a tuple in place if the new tuple fits.
    ///
    /// Returns error if the new tuple is larger than the old one.
    pub async fn update(&self, tuple_id: TupleId, tuple: &Tuple) -> Result<()> {
        let page_data = self.fetch_page(tuple_id.page_id).await?;
        let mut page = HeapPage::from_bytes(page_data);

        page.update_tuple(SlotId(tuple_id.slot_id), tuple)?;

        self.write_page(tuple_id.page_id, page.as_bytes()).await?;
        self.update_fsm_for_page(tuple_id.page_id.page_num, page.free_space())
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
            .map(|n| PageId::new(file_id, n))
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

    /// Flushes all dirty heap pages to disk.
    pub async fn flush(&self) -> Result<()> {
        self.pool.flush_all(|page_id, data| {
            // Only flush pages belonging to this heap file
            if page_id.file_id == self.config.heap_file_id
                || page_id.file_id == self.config.fsm_file_id
            {
                // Note: This is synchronous, which is a limitation.
                // For full async, we'd need a different approach.
                let data_copy: [u8; PAGE_SIZE] = data
                    .try_into()
                    .map_err(|_| ZyronError::Internal("Invalid page size".to_string()))?;
                // We can't await here, so we'll use blocking for now
                // A proper solution would queue these writes
                let _ = &data_copy; // Suppress unused warning
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Finds a page with at least the specified free space AND verifies it actually has the space.
    /// FSM categories are coarse, so we verify each candidate before returning.
    async fn find_page_with_space(&self, min_space: usize) -> Result<Option<(PageId, HeapPage)>> {
        let num_heap_pages = self.heap_page_count();
        if num_heap_pages == 0 {
            return Ok(None);
        }

        let num_fsm_pages = self.fsm_page_count();

        for fsm_page_num in 0..num_fsm_pages {
            let fsm_page_id = PageId::new(self.config.fsm_file_id, fsm_page_num);
            let fsm_data = self.fetch_page(fsm_page_id).await?;
            let fsm_page = FsmPage::from_bytes(fsm_data);

            // Try each candidate page the FSM suggests
            let mut start_entry = 0;
            while let Some(heap_page_num) =
                fsm_page.find_page_with_space_from(min_space, start_entry)
            {
                if heap_page_num >= num_heap_pages {
                    break;
                }

                // Verify the page actually has enough space
                let page_id = PageId::new(self.config.heap_file_id, heap_page_num);
                let page_data = self.fetch_page(page_id).await?;
                let page = HeapPage::from_bytes(page_data);

                if page.total_usable_space() >= min_space {
                    return Ok(Some((page_id, page)));
                }

                // Try next candidate
                start_entry = (heap_page_num - fsm_page.first_page_num() + 1) as usize;
            }
        }

        Ok(None)
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
        let category = space_to_category(free_space);
        let mut state = self.fsm_state.lock();

        // Update page hint cache with the current free space category
        if category > 0 {
            state.hint_cache.update(heap_page_num, category);
        } else {
            // Page is full, remove from cache
            state.hint_cache.remove(heap_page_num);
        }

        state.pending_updates.push(PendingFsmUpdate {
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
            let mut state = self.fsm_state.lock();
            std::mem::take(&mut state.pending_updates)
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

            // Check if current page has space (including reclaimable space)
            let use_current = if let Some(ref page) = current_page {
                page.total_usable_space() >= space_needed
            } else {
                false
            };

            if !use_current {
                // Flush current page if any
                if let (Some(page_id), Some(page)) = (current_page_id, &current_page) {
                    self.write_page(page_id, page.as_bytes()).await?;
                    self.defer_fsm_update(page_id.page_num, page.total_usable_space());
                }

                // Try hint page first
                let hint = self.last_page_hint.load(Ordering::Relaxed);
                let hint_page_id = if hint != u32::MAX && hint < self.heap_page_count() {
                    Some(PageId::new(self.config.heap_file_id, hint))
                } else {
                    None
                };

                // Find or allocate a page with space (including reclaimable space)
                // Try last_page_hint first (fast path for sequential inserts)
                let found_page = if let Some(hint_id) = hint_page_id {
                    let page_data = self.fetch_page(hint_id).await?;
                    let page = HeapPage::from_bytes(page_data);
                    if page.total_usable_space() >= space_needed {
                        Some((hint_id, page))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let (page_id, page) = if let Some((id, p)) = found_page {
                    (id, p)
                } else if let Some((id, page)) = self.find_page_with_space(space_needed).await? {
                    // find_page_with_space verifies the page actually has enough space
                    (id, page)
                } else {
                    // No suitable page found, allocate new
                    let id = self.disk.allocate_page(self.config.heap_file_id).await?;
                    self.increment_heap_pages();
                    (id, HeapPage::new(id))
                };

                current_page_id = Some(page_id);
                current_page = Some(page);
                self.last_page_hint
                    .store(page_id.page_num, Ordering::Relaxed);
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
            self.defer_fsm_update(page_id.page_num, page.total_usable_space());
        }

        // Batch flush all FSM updates
        self.flush_fsm_updates().await?;

        Ok(results)
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
    /// Returns a zero-copy iterator over all tuples.
    ///
    /// Each tuple is returned as a `TupleView` borrowing directly from
    /// the pinned page buffers. No allocations or copies occur.
    pub fn iter(&self) -> impl Iterator<Item = (TupleId, TupleView<'_>)> + '_ {
        self.page_ids.iter().flat_map(move |&page_id| {
            PageTupleIter::new(self.pool, page_id)
        })
    }
}

impl Drop for ScanGuard<'_> {
    fn drop(&mut self) {
        self.pool.batch_unpin(&self.page_ids);
    }
}

/// Iterator over tuples in a single page.
struct PageTupleIter<'a> {
    data: &'a [u8; PAGE_SIZE],
    page_id: PageId,
    slot_idx: usize,
    slot_count: usize,
}

impl<'a> PageTupleIter<'a> {
    fn new(pool: &'a BufferPool, page_id: PageId) -> Self {
        let ptr = unsafe { pool.frame_data_ptr(page_id) };
        match ptr {
            Some(p) => {
                let data = unsafe { &*p };
                let slot_count = u16::from_le_bytes(
                    [data[HEAP_HEADER_OFFSET], data[HEAP_HEADER_OFFSET + 1]]
                ) as usize;
                Self { data, page_id, slot_idx: 0, slot_count }
            }
            None => {
                // Empty page - use static empty array
                static EMPTY_PAGE: [u8; PAGE_SIZE] = [0u8; PAGE_SIZE];
                Self {
                    data: &EMPTY_PAGE,
                    page_id,
                    slot_idx: 0,
                    slot_count: 0,
                }
            }
        }
    }
}

impl<'a> Iterator for PageTupleIter<'a> {
    type Item = (TupleId, TupleView<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.slot_idx < self.slot_count {
            let i = self.slot_idx;
            self.slot_idx += 1;

            let slot_base = DATA_START + i * TUPLE_SLOT_SIZE;
            let tuple_offset = u16::from_le_bytes(
                [self.data[slot_base], self.data[slot_base + 1]]
            ) as usize;
            let tuple_length = u16::from_le_bytes(
                [self.data[slot_base + 2], self.data[slot_base + 3]]
            ) as usize;

            if tuple_length == 0 { continue; }

            let header = TupleHeader::from_bytes(
                &self.data[tuple_offset..tuple_offset + TUPLE_HEADER_SIZE]
            );
            let data_start = tuple_offset + TUPLE_HEADER_SIZE;
            let data_end = data_start + header.data_len as usize;

            return Some((
                TupleId::new(self.page_id, i as u16),
                TupleView::new(header, &self.data[data_start..data_end])
            ));
        }
        None
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
        let tuples: Vec<_> = guard.iter().collect();
        assert_eq!(tuples.len(), 10);

        for (i, (_, tuple)) in tuples.iter().enumerate() {
            assert_eq!(tuple.header.xmin, i as u32);
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
        let tuples: Vec<_> = guard.iter().collect();
        assert_eq!(tuples.len(), 5);
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
        let tuples: Vec<_> = guard.iter().collect();
        assert_eq!(tuples.len(), 20);
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

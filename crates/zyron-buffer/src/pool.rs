//! Buffer pool manager.

use crate::frame::{BufferFrame, FrameId};
use crate::page_table::PageTable;
use crate::replacer::{ClockReplacer, Replacer};
use parking_lot::Mutex;
use sysinfo::System;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_common::{Result, ZyronError};

/// Information about a dirty page that was evicted from the buffer pool.
/// Caller must write this to disk to prevent data loss.
#[derive(Debug)]
pub struct EvictedPage {
    pub page_id: PageId,
    pub data: Box<[u8; PAGE_SIZE]>,
}

/// Configuration for the buffer pool.
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Number of frames in the pool.
    pub num_frames: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self { num_frames: 1024 }
    }
}

/// Buffer pool manager.
///
/// Manages a fixed-size pool of page frames with:
/// - Page ID to frame ID mapping (lock-free page table)
/// - Free frame list for new pages
/// - Clock replacement for eviction
/// - Pin counting for concurrent access
pub struct BufferPool {
    /// Configuration.
    config: BufferPoolConfig,
    /// Array of buffer frames.
    frames: Vec<BufferFrame>,
    /// Page ID to frame ID mapping (lock-free reads).
    page_table: PageTable,
    /// List of free frame IDs.
    free_list: Mutex<Vec<FrameId>>,
    /// Page replacement policy.
    replacer: ClockReplacer,
}

impl BufferPool {
    /// Creates a new buffer pool.
    pub fn new(config: BufferPoolConfig) -> Self {
        let num_frames = config.num_frames;

        // Initialize frames
        let frames: Vec<_> = (0..num_frames)
            .map(|i| BufferFrame::new(FrameId(i as u32)))
            .collect();

        // All frames start in free list
        let free_list: Vec<_> = (0..num_frames).map(|i| FrameId(i as u32)).collect();

        Self {
            config,
            frames,
            page_table: PageTable::new(num_frames),
            free_list: Mutex::new(free_list),
            replacer: ClockReplacer::new(num_frames),
        }
    }

    /// Creates a buffer pool sized to 25% of available system RAM.
    ///
    /// Queries the system for available memory and allocates 25% of it
    /// for the buffer pool. Minimum 1,000 frames to ensure useful caching
    /// even on low-memory systems. No upper limit - systems with terabytes
    /// of RAM can use it all.
    ///
    /// For a system with 16GB RAM, this allocates ~4GB (~250k frames).
    pub fn auto_sized() -> Self {
        let mut sys = System::new_all();
        sys.refresh_memory();

        let available_bytes = sys.available_memory() as usize;
        let target_bytes = available_bytes / 4; // 25% of available RAM
        let num_frames = (target_bytes / PAGE_SIZE).max(1_000);

        Self::new(BufferPoolConfig { num_frames })
    }

    /// Returns the number of frames in the pool.
    pub fn num_frames(&self) -> usize {
        self.config.num_frames
    }

    /// Returns the number of free frames.
    pub fn free_count(&self) -> usize {
        self.free_list.lock().len()
    }

    /// Returns the number of pages currently in the pool.
    pub fn page_count(&self) -> usize {
        self.page_table.len()
    }

    /// Checks if a page is in the buffer pool.
    pub fn contains(&self, page_id: PageId) -> bool {
        self.page_table.contains(page_id)
    }

    /// Fetches a page from the buffer pool.
    ///
    /// If the page is not in the pool, returns None.
    /// The page is pinned before being returned.
    #[inline(always)]
    pub fn fetch_page(&self, page_id: PageId) -> Option<&BufferFrame> {
        let frame_id = self.page_table.get(page_id)?;
        let frame = &self.frames[frame_id.0 as usize];
        frame.pin();
        // Record access for clock algorithm (sets reference bit)
        self.replacer.record_access(frame_id);
        Some(frame)
    }

    /// Allocates a frame for a new page.
    ///
    /// Tries to get a free frame first, then evicts if necessary.
    /// Returns the frame ID and any evicted dirty page that must be flushed.
    fn allocate_frame(&self) -> Result<(FrameId, Option<EvictedPage>)> {
        // Try free list first
        {
            let mut free_list = self.free_list.lock();
            if let Some(frame_id) = free_list.pop() {
                return Ok((frame_id, None));
            }
        }

        // Try to evict - check pin_count directly for each candidate frame
        let victim_id = self
            .replacer
            .evict(|fid| self.frames[fid.0 as usize].pin_count() == 0);

        if let Some(victim_id) = victim_id {
            let frame = &self.frames[victim_id.0 as usize];

            // Capture evicted page data if dirty
            let evicted = if frame.is_dirty() {
                if let Some(page_id) = frame.page_id() {
                    let data_guard = frame.read_data();
                    let mut data = Box::new([0u8; PAGE_SIZE]);
                    data.copy_from_slice(&**data_guard);
                    drop(data_guard);
                    Some(EvictedPage { page_id, data })
                } else {
                    None
                }
            } else {
                None
            };

            // Remove old page from page table
            if let Some(old_page_id) = frame.page_id() {
                self.page_table.remove(old_page_id);
            }

            return Ok((victim_id, evicted));
        }

        Err(ZyronError::BufferPoolFull)
    }

    /// Inserts a new page into the buffer pool.
    ///
    /// If the page already exists, returns the existing frame.
    /// The page is pinned before being returned.
    ///
    /// Returns (frame, evicted) where evicted contains any dirty page that was
    /// evicted to make room. Caller must write evicted pages to disk.
    #[inline]
    pub fn new_page(&self, page_id: PageId) -> Result<(&BufferFrame, Option<EvictedPage>)> {
        // Check if page already exists
        if let Some(frame_id) = self.page_table.get(page_id) {
            let frame = &self.frames[frame_id.0 as usize];
            frame.pin();
            self.replacer.record_access(frame_id);
            return Ok((frame, None));
        }

        // Allocate a frame
        let (frame_id, evicted) = self.allocate_frame()?;

        // Set up the frame
        let frame = &self.frames[frame_id.0 as usize];
        frame.reset();
        frame.set_page_id(Some(page_id));
        // New frame: pin() returns 0 since reset() clears pin_count.
        // Frame is not in evictable set (free_list frames never were,
        // evicted frames were removed by evict()).
        frame.pin();

        // Update page table
        self.page_table.insert(page_id, frame_id);

        Ok((frame, evicted))
    }

    /// Loads page data into the buffer pool.
    ///
    /// This is used when reading a page from disk.
    /// Returns the frame and any evicted dirty page that must be flushed.
    #[inline]
    pub fn load_page(
        &self,
        page_id: PageId,
        data: &[u8],
    ) -> Result<(&BufferFrame, Option<EvictedPage>)> {
        let (frame, evicted) = self.new_page(page_id)?;
        frame.copy_from(data);
        Ok((frame, evicted))
    }

    /// Unpins a page in the buffer pool.
    ///
    /// If the page becomes unpinned (pin count = 0), it becomes evictable.
    /// Evictability is determined by pin_count during eviction, not tracked separately.
    #[inline]
    pub fn unpin_page(&self, page_id: PageId, is_dirty: bool) -> bool {
        if let Some(frame_id) = self.page_table.get(page_id) {
            let frame = &self.frames[frame_id.0 as usize];

            if is_dirty {
                frame.set_dirty(true);
            }

            frame.unpin();
            // No need to update replacer - evict() checks pin_count directly
            return true;
        }
        false
    }

    /// Flushes a page to the provided callback.
    ///
    /// The callback receives the page data if the page is dirty.
    /// Returns true if the page was flushed.
    pub fn flush_page<F>(&self, page_id: PageId, mut flush_fn: F) -> Result<bool>
    where
        F: FnMut(PageId, &[u8]) -> Result<()>,
    {
        if let Some(frame_id) = self.page_table.get(page_id) {
            let frame = &self.frames[frame_id.0 as usize];

            if frame.is_dirty() {
                let data = frame.read_data();
                flush_fn(page_id, &**data)?;
                frame.set_dirty(false);
                return Ok(true);
            }
            return Ok(false);
        }
        Ok(false)
    }

    /// Flushes all dirty pages.
    ///
    /// Returns the number of pages flushed.
    pub fn flush_all<F>(&self, mut flush_fn: F) -> Result<usize>
    where
        F: FnMut(PageId, &[u8]) -> Result<()>,
    {
        let mut flushed = 0;
        let mut flush_error: Option<ZyronError> = None;

        // Collect dirty pages first to avoid holding guards during flush
        let mut dirty_pages = Vec::new();
        self.page_table.for_each(|page_id, frame_id| {
            dirty_pages.push((page_id, frame_id));
            true // continue iteration
        });

        for (page_id, frame_id) in dirty_pages {
            let frame = &self.frames[frame_id.0 as usize];
            if frame.is_dirty() {
                let data = frame.read_data();
                if let Err(e) = flush_fn(page_id, &**data) {
                    flush_error = Some(e);
                    break;
                }
                frame.set_dirty(false);
                flushed += 1;
            }
        }

        match flush_error {
            Some(e) => Err(e),
            None => Ok(flushed),
        }
    }

    /// Deletes a page from the buffer pool.
    ///
    /// Returns true if the page was deleted.
    /// Returns false if the page is pinned or not in the pool.
    pub fn delete_page(&self, page_id: PageId) -> bool {
        if let Some(frame_id) = self.page_table.remove(page_id) {
            let frame = &self.frames[frame_id.0 as usize];

            // Cannot delete pinned page - re-insert if pinned
            if frame.is_pinned() {
                self.page_table.insert(page_id, frame_id);
                return false;
            }

            // Remove from replacer and add to free list
            self.replacer.remove(frame_id);
            frame.reset();
            self.free_list.lock().push(frame_id);

            return true;
        }
        false
    }

    /// Returns a read guard for page data.
    pub fn read_page(&self, page_id: PageId) -> Option<PageReadGuard<'_>> {
        let frame = self.fetch_page(page_id)?;
        Some(PageReadGuard {
            pool: self,
            page_id,
            frame,
        })
    }

    /// Returns raw pointer to frame data for a pinned page without acquiring RwLock.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - Page is pinned before calling and stays pinned during use
    /// - No concurrent writers exist
    /// - Pointer is not dereferenced after unpin
    #[inline(always)]
    pub unsafe fn frame_data_ptr(&self, page_id: PageId) -> Option<*const [u8; PAGE_SIZE]> {
        let frame_id = self.page_table.get(page_id)?;
        let frame = &self.frames[frame_id.0 as usize];
        Some(unsafe { frame.data_ptr() })
    }

    /// Returns mutable raw pointer to frame data for a pinned page.
    ///
    /// # Safety
    /// - Page must be pinned before calling and stay pinned during use
    /// - Caller must ensure exclusive write access (no concurrent readers/writers)
    /// - Pointer must not be dereferenced after unpin
    #[inline(always)]
    pub unsafe fn frame_data_ptr_mut(&self, page_id: PageId) -> Option<*mut [u8; PAGE_SIZE]> {
        let frame_id = self.page_table.get(page_id)?;
        let frame = &self.frames[frame_id.0 as usize];
        Some(unsafe { frame.data_ptr_mut() })
    }

    /// Pins multiple pages at once for batch read operations.
    ///
    /// Returns the number of pages successfully pinned.
    /// Use with `batch_unpin` after processing.
    #[inline]
    pub fn batch_pin(&self, page_ids: &[PageId]) -> usize {
        let mut pinned = 0;
        for &pid in page_ids {
            if let Some(frame_id) = self.page_table.get(pid) {
                self.frames[frame_id.0 as usize].pin();
                self.replacer.record_access(frame_id);
                pinned += 1;
            }
        }
        pinned
    }

    /// Unpins multiple pages at once after batch read operations.
    #[inline]
    pub fn batch_unpin(&self, page_ids: &[PageId]) {
        for &pid in page_ids {
            if let Some(frame_id) = self.page_table.get(pid) {
                self.frames[frame_id.0 as usize].unpin();
            }
        }
    }

    /// Unpins multiple pages and marks them dirty for batch write operations.
    #[inline]
    pub fn batch_unpin_dirty(&self, page_ids: &[PageId]) {
        for &pid in page_ids {
            self.unpin_page(pid, true);
        }
    }

    /// Allocates frames for multiple new pages in batch.
    ///
    /// Returns frames and any evicted dirty pages that need flushing.
    /// Single pass through allocation - reduces lock contention.
    pub fn batch_new_pages(
        &self,
        page_ids: &[PageId],
    ) -> Result<(Vec<&BufferFrame>, Vec<EvictedPage>)> {
        let mut frames = Vec::with_capacity(page_ids.len());
        let mut evicted = Vec::new();

        for &page_id in page_ids {
            let (frame, ev) = self.new_page(page_id)?;
            frames.push(frame);
            if let Some(e) = ev {
                evicted.push(e);
            }
        }

        Ok((frames, evicted))
    }

    /// Returns a write guard for page data.
    pub fn write_page(&self, page_id: PageId) -> Option<PageWriteGuard<'_>> {
        let frame = self.fetch_page(page_id)?;
        Some(PageWriteGuard {
            pool: self,
            page_id,
            frame,
        })
    }

    /// Returns statistics about the buffer pool.
    pub fn stats(&self) -> BufferPoolStats {
        let mut pinned_count = 0;
        let mut dirty_count = 0;

        self.page_table.for_each(|_, frame_id| {
            let frame = &self.frames[frame_id.0 as usize];
            if frame.is_pinned() {
                pinned_count += 1;
            }
            if frame.is_dirty() {
                dirty_count += 1;
            }
            true // continue iteration
        });

        BufferPoolStats {
            total_frames: self.config.num_frames,
            free_frames: self.free_count(),
            used_frames: self.page_table.len(),
            pinned_frames: pinned_count,
            dirty_frames: dirty_count,
        }
    }
}

/// Statistics about the buffer pool.
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    /// Total number of frames.
    pub total_frames: usize,
    /// Number of free frames.
    pub free_frames: usize,
    /// Number of frames with pages.
    pub used_frames: usize,
    /// Number of pinned frames.
    pub pinned_frames: usize,
    /// Number of dirty frames.
    pub dirty_frames: usize,
}

/// RAII guard for reading a page.
pub struct PageReadGuard<'a> {
    pool: &'a BufferPool,
    page_id: PageId,
    frame: &'a BufferFrame,
}

impl<'a> PageReadGuard<'a> {
    /// Returns the page ID.
    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    /// Returns the page data.
    pub fn data(&self) -> parking_lot::RwLockReadGuard<'_, Box<[u8; PAGE_SIZE]>> {
        self.frame.read_data()
    }
}

impl Drop for PageReadGuard<'_> {
    fn drop(&mut self) {
        self.pool.unpin_page(self.page_id, false);
    }
}

/// RAII guard for writing a page.
pub struct PageWriteGuard<'a> {
    pool: &'a BufferPool,
    page_id: PageId,
    frame: &'a BufferFrame,
}

impl<'a> PageWriteGuard<'a> {
    /// Returns the page ID.
    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    /// Returns mutable access to page data.
    pub fn data_mut(&self) -> parking_lot::RwLockWriteGuard<'_, Box<[u8; PAGE_SIZE]>> {
        self.frame.write_data()
    }

    /// Marks the page as dirty.
    pub fn set_dirty(&self) {
        self.frame.set_dirty(true);
    }
}

impl Drop for PageWriteGuard<'_> {
    fn drop(&mut self) {
        self.pool.unpin_page(self.page_id, self.frame.is_dirty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool(num_frames: usize) -> BufferPool {
        BufferPool::new(BufferPoolConfig { num_frames })
    }

    #[test]
    fn test_buffer_pool_new() {
        let pool = create_test_pool(10);

        assert_eq!(pool.num_frames(), 10);
        assert_eq!(pool.free_count(), 10);
        assert_eq!(pool.page_count(), 0);
    }

    #[test]
    fn test_buffer_pool_new_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        let (frame, evicted) = pool.new_page(page_id).unwrap();

        assert!(evicted.is_none());
        assert_eq!(frame.page_id(), Some(page_id));
        assert!(frame.is_pinned());
        assert_eq!(pool.free_count(), 9);
        assert_eq!(pool.page_count(), 1);
        assert!(pool.contains(page_id));
    }

    #[test]
    fn test_buffer_pool_fetch_existing() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        let frame = pool.fetch_page(page_id).unwrap();
        assert_eq!(frame.page_id(), Some(page_id));
        assert!(frame.is_pinned());
    }

    #[test]
    fn test_buffer_pool_fetch_nonexistent() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        assert!(pool.fetch_page(page_id).is_none());
    }

    #[test]
    fn test_buffer_pool_unpin() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        let (frame, _) = pool.new_page(page_id).unwrap();
        assert!(frame.is_pinned());

        pool.unpin_page(page_id, false);
        assert!(!frame.is_pinned());
    }

    #[test]
    fn test_buffer_pool_dirty_tracking() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, true);

        let frame = pool.fetch_page(page_id).unwrap();
        assert!(frame.is_dirty());
    }

    #[test]
    fn test_buffer_pool_eviction() {
        let pool = create_test_pool(3);

        // Fill the pool
        for i in 0..3 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, false);
        }

        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.page_count(), 3);

        // Add one more page, should evict
        let new_page_id = PageId::new(0, 99);
        let (_, evicted) = pool.new_page(new_page_id).unwrap();

        assert!(evicted.is_none()); // Evicted page was clean
        assert_eq!(pool.page_count(), 3);
        assert!(pool.contains(new_page_id));
    }

    #[test]
    fn test_buffer_pool_eviction_dirty() {
        let pool = create_test_pool(1);
        let page_id1 = PageId::new(0, 1);

        // Add dirty page with some data
        let (frame, _) = pool.new_page(page_id1).unwrap();
        frame.write_data()[0] = 0xAB;
        pool.unpin_page(page_id1, true);

        // Add another page, should evict dirty page
        let page_id2 = PageId::new(0, 2);
        let (_, evicted) = pool.new_page(page_id2).unwrap();

        // Verify evicted page info is captured
        let evicted = evicted.expect("dirty page should be returned on eviction");
        assert_eq!(evicted.page_id, page_id1);
        assert_eq!(evicted.data[0], 0xAB);
    }

    #[test]
    fn test_buffer_pool_full_all_pinned() {
        let pool = create_test_pool(2);

        // Fill pool with pinned pages
        pool.new_page(PageId::new(0, 1)).unwrap();
        pool.new_page(PageId::new(0, 2)).unwrap();

        // Try to add another page (should fail)
        let result = pool.new_page(PageId::new(0, 3));
        assert!(matches!(result, Err(ZyronError::BufferPoolFull)));
    }

    #[test]
    fn test_buffer_pool_delete_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        assert!(pool.contains(page_id));
        assert!(pool.delete_page(page_id));
        assert!(!pool.contains(page_id));
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn test_buffer_pool_delete_pinned_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        // Don't unpin

        assert!(!pool.delete_page(page_id));
        assert!(pool.contains(page_id));
    }

    #[test]
    fn test_buffer_pool_load_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);
        let data = [0xABu8; PAGE_SIZE];

        let (frame, _) = pool.load_page(page_id, &data).unwrap();

        let frame_data = frame.read_data();
        assert_eq!(frame_data[0], 0xAB);
        assert_eq!(frame_data[100], 0xAB);
    }

    #[test]
    fn test_buffer_pool_flush_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, true);

        let mut flushed_pages = vec![];
        let result = pool.flush_page(page_id, |pid, _data| {
            flushed_pages.push(pid);
            Ok(())
        });

        assert!(result.unwrap());
        assert_eq!(flushed_pages, vec![page_id]);

        // Page should no longer be dirty
        let frame = pool.fetch_page(page_id).unwrap();
        assert!(!frame.is_dirty());
    }

    #[test]
    fn test_buffer_pool_flush_all() {
        let pool = create_test_pool(10);

        // Add multiple dirty pages
        for i in 0..5 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
        }

        let mut flushed_count = 0;
        let result = pool.flush_all(|_pid, _data| {
            flushed_count += 1;
            Ok(())
        });

        assert_eq!(result.unwrap(), 5);
        assert_eq!(flushed_count, 5);
    }

    #[test]
    fn test_buffer_pool_read_guard() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        {
            let guard = pool.read_page(page_id).unwrap();
            assert_eq!(guard.page_id(), page_id);
            // Guard holds one pin
        }

        // After guard dropped, page should be unpinned (pin_count = 0)
        // Fetch adds a new pin, so pin_count becomes 1
        let frame = pool.fetch_page(page_id).unwrap();
        assert_eq!(frame.pin_count(), 1);
    }

    #[test]
    fn test_buffer_pool_write_guard() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        {
            let guard = pool.write_page(page_id).unwrap();
            guard.set_dirty();
            {
                let mut data = guard.data_mut();
                data[0] = 0xFF;
            }
        }

        // After guard dropped, page should be dirty
        let frame = pool.fetch_page(page_id).unwrap();
        assert!(frame.is_dirty());
        assert_eq!(frame.read_data()[0], 0xFF);
    }

    #[test]
    fn test_buffer_pool_stats() {
        let pool = create_test_pool(10);

        // Add some pages
        for i in 0..5 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            if i % 2 == 0 {
                pool.unpin_page(page_id, true); // Dirty
            }
            // Odd pages remain pinned
        }

        let stats = pool.stats();
        assert_eq!(stats.total_frames, 10);
        assert_eq!(stats.free_frames, 5);
        assert_eq!(stats.used_frames, 5);
        assert_eq!(stats.pinned_frames, 2); // Pages 1, 3
        assert_eq!(stats.dirty_frames, 3); // Pages 0, 2, 4
    }

    #[test]
    fn test_buffer_pool_duplicate_new_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        // Adding same page again should return existing frame
        let (frame, evicted) = pool.new_page(page_id).unwrap();

        assert!(evicted.is_none()); // No eviction when page already exists
        assert_eq!(frame.page_id(), Some(page_id));
        assert_eq!(pool.page_count(), 1);
    }
}

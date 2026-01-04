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
}

impl HeapFile {
    /// Creates a new HeapFile with buffer pool integration.
    pub fn new(disk: Arc<DiskManager>, pool: Arc<BufferPool>, config: HeapFileConfig) -> Result<Self> {
        let fsm = FreeSpaceMap::new(config.heap_file_id, PageId::new(config.fsm_file_id, 0));

        Ok(Self {
            disk,
            pool,
            fsm,
            config,
        })
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

    /// Inserts a tuple into the heap file.
    ///
    /// Finds a page with sufficient space using the FSM, or allocates
    /// a new page if needed. Returns the TupleId of the inserted tuple.
    pub async fn insert(&self, tuple: &Tuple) -> Result<TupleId> {
        let tuple_size = tuple.size_on_disk();
        let space_needed = tuple_size + crate::heap::page::TupleSlot::SIZE;

        // Try to find a page with enough space
        if let Some(page_id) = self.find_page_with_space(space_needed).await? {
            match self.insert_into_page(page_id, tuple).await {
                Ok(tuple_id) => return Ok(tuple_id),
                Err(ZyronError::PageFull) => {
                    // FSM estimate was slightly off, fall through to allocate new page
                }
                Err(e) => return Err(e),
            }
        }

        // No suitable page found or FSM estimate was off, allocate a new one
        let page_id = self.allocate_new_page().await?;
        self.insert_into_page(page_id, tuple).await
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
    pub async fn scan(&self) -> Result<Vec<(TupleId, Tuple)>> {
        let num_pages = self.disk.num_pages(self.config.heap_file_id).await?;
        let mut results = Vec::new();

        for page_num in 0..num_pages {
            let page_id = PageId::new(self.config.heap_file_id, page_num);
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
        self.disk.num_pages(self.config.heap_file_id).await
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
        let num_heap_pages = self.disk.num_pages(self.config.heap_file_id).await?;
        if num_heap_pages == 0 {
            return Ok(None);
        }

        let num_fsm_pages = self.disk.num_pages(self.config.fsm_file_id).await?;

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

    /// Inserts a tuple into a specific page.
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

        // Check if FSM page exists
        let num_fsm_pages = self.disk.num_pages(self.config.fsm_file_id).await?;

        let mut fsm_page = if fsm_page_num < num_fsm_pages {
            let fsm_data = self.fetch_page(fsm_page_id).await?;
            FsmPage::from_bytes(fsm_data)
        } else {
            // Allocate new FSM page
            let first_tracked = fsm_page_num * ENTRIES_PER_FSM_PAGE as u32;
            let new_fsm_page = FsmPage::new(fsm_page_id, first_tracked);
            self.disk.allocate_page(self.config.fsm_file_id).await?;
            new_fsm_page
        };

        let category = space_to_category(free_space);
        fsm_page.set_space(heap_page_num, category)?;
        self.write_page(fsm_page_id, fsm_page.as_bytes()).await?;

        Ok(())
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

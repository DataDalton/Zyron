//! Page-based B+Tree index implementation.

use bytes::Bytes;
use parking_lot::RwLock;
use std::sync::Arc;
use crate::disk::DiskManager;
use crate::tuple::TupleId;
use super::constants::MAX_KEY_SIZE;
use super::page::{BTreeInternalPage, BTreeLeafPage};
use super::store::InMemoryPageStore;
use super::types::{compare_keys, DeleteResult};
use zyron_buffer::BufferPool;
use zyron_common::page::PageId;
use zyron_common::{Result, ZyronError};

pub struct BTreeIndex {
    /// In-memory page storage (all nodes stored here).
    pages: RwLock<InMemoryPageStore>,
    /// Root page number (index into pages).
    root_page_num: std::sync::atomic::AtomicU32,
    /// Tree height (1 = just root as leaf).
    height: std::sync::atomic::AtomicU32,
    /// File ID for this index (used for PageId construction).
    file_id: u32,
    /// Disk manager for checkpoint I/O (not used in hot path).
    #[allow(dead_code)]
    disk: Arc<DiskManager>,
    /// Buffer pool reference (kept for compatibility, not used in hot path).
    #[allow(dead_code)]
    pool: Arc<BufferPool>,
}

impl BTreeIndex {
    /// Maximum B+Tree height (supports billions of keys).
    const MAX_HEIGHT: usize = 16;

    /// Creates a new B+ tree index with fully in-memory storage.
    ///
    /// All pages are stored in RAM. Disk/BufferPool are kept for compatibility
    /// but not used in the hot path.
    pub async fn create(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
    ) -> Result<Self> {
        use std::sync::atomic::AtomicU32;

        // Create in-memory page store
        let mut store = InMemoryPageStore::new();

        // Allocate root leaf page (page 0)
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num);

        // Initialize root as empty leaf
        let root_page = BTreeLeafPage::new(root_page_id);
        store.write(root_page_num, root_page.as_bytes());

        Ok(Self {
            pages: RwLock::new(store),
            root_page_num: AtomicU32::new(root_page_num),
            height: AtomicU32::new(1),
            file_id,
            disk,
            pool,
        })
    }

    /// Opens an existing B+ tree index (placeholder - would load from checkpoint).
    pub async fn open(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
        _root_page_id: PageId,
        height: u32,
    ) -> Result<Self> {
        use std::sync::atomic::AtomicU32;

        // Create empty in-memory store (in production, would load from checkpoint)
        let mut store = InMemoryPageStore::new();
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num);
        let root_page = BTreeLeafPage::new(root_page_id);
        store.write(root_page_num, root_page.as_bytes());

        Ok(Self {
            pages: RwLock::new(store),
            root_page_num: AtomicU32::new(root_page_num),
            height: AtomicU32::new(height),
            file_id,
            disk,
            pool,
        })
    }

    /// Returns the root page ID.
    #[inline]
    pub fn root_page_id(&self) -> PageId {
        PageId::new(
            self.file_id,
            self.root_page_num
                .load(std::sync::atomic::Ordering::Acquire),
        )
    }

    /// Returns the file ID.
    #[inline]
    pub fn file_id(&self) -> u32 {
        self.file_id
    }

    /// Returns the tree height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height.load(std::sync::atomic::Ordering::Acquire)
    }

    // =========================================================================
    // Core In-Memory Operations (Synchronous)
    // =========================================================================

    /// Finds the leaf page number for a given key using existing pages reference.
    #[inline]
    fn find_leaf_in_pages(&self, pages: &InMemoryPageStore, key: &[u8]) -> u32 {
        let height = self.height.load(std::sync::atomic::Ordering::Relaxed);
        let mut current = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Relaxed);

        // Height 1 means root is the leaf
        if height == 1 {
            return current;
        }

        // Traverse internal nodes
        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num;
            } else {
                break;
            }
        }

        current
    }

    /// Finds the leaf page number for a given key (takes read lock).
    #[inline]
    fn find_leaf_page_num(&self, key: &[u8]) -> u32 {
        let pages = self.pages.read();
        self.find_leaf_in_pages(&pages, key)
    }

    /// Searches for a key synchronously. All data is in RAM.
    /// Single lock acquisition for entire operation.
    #[inline]
    pub fn search_sync(&self, key: &[u8]) -> Option<TupleId> {
        let pages = self.pages.read();
        let leaf_page_num = self.find_leaf_in_pages(&pages, key);

        if let Some(data) = pages.get(leaf_page_num) {
            BTreeLeafPage::get_in_slice(data, key)
        } else {
            None
        }
    }

    /// Inserts a key-value pair synchronously. All data is in RAM.
    /// Single lock acquisition for entire operation (no split case).
    #[inline]
    pub fn insert_sync(&self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        if key.len() > MAX_KEY_SIZE {
            return Err(ZyronError::KeyTooLarge {
                size: key.len(),
                max: MAX_KEY_SIZE,
            });
        }

        // Fast path: single write lock for find + insert
        {
            let mut pages = self.pages.write();
            let leaf_page_num = self.find_leaf_in_pages(&pages, key);

            if let Some(data) = pages.get_mut(leaf_page_num) {
                match BTreeLeafPage::insert_in_slice(data, key, tuple_id) {
                    Ok(()) => return Ok(()),
                    Err(ZyronError::NodeFull) => {
                        // Need split - fall through
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Slow path: need to split
        self.insert_with_split_sync(Bytes::copy_from_slice(key), tuple_id)
    }

    /// Inserts a key-value pair with exclusive access (no locking).
    /// Use when caller has &mut BTreeIndex for maximum performance.
    /// Fully inlined fast path for minimal overhead.
    #[inline(always)]
    pub fn insert_exclusive(&mut self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        if key.len() > MAX_KEY_SIZE {
            return Err(ZyronError::KeyTooLarge {
                size: key.len(),
                max: MAX_KEY_SIZE,
            });
        }

        let pages = self.pages.get_mut();
        let height = *self.height.get_mut();
        let root = *self.root_page_num.get_mut();

        // Inline leaf finding for fast path
        let mut current = root;
        let mut path = [0u32; Self::MAX_HEIGHT];
        let mut path_len = 0;

        path[path_len] = current;
        path_len += 1;

        if height > 1 {
            for _ in 0..(height - 1) {
                if let Some(data) = pages.get(current) {
                    let child = BTreeInternalPage::find_child_in_slice(data, key);
                    current = child.page_num;
                    path[path_len] = current;
                    path_len += 1;
                } else {
                    return Err(ZyronError::BTreeCorrupted(
                        "internal node not found".to_string(),
                    ));
                }
            }
        }

        // Try insert in leaf (fast path - most common)
        if let Some(data) = pages.get_mut(current) {
            match BTreeLeafPage::insert_in_slice(data, key, tuple_id) {
                Ok(()) => return Ok(()),
                Err(ZyronError::NodeFull) => {
                    // Fall through to split path
                }
                Err(e) => return Err(e),
            }
        } else {
            return Err(ZyronError::BTreeCorrupted("leaf not found".to_string()));
        }

        // Split handling (rare path)
        self.insert_with_split_exclusive(Bytes::copy_from_slice(key), tuple_id, &path[..path_len])
    }

    /// Insert with split using exclusive access (no locking).
    fn insert_with_split_exclusive(
        &mut self,
        key: Bytes,
        tuple_id: TupleId,
        path: &[u32],
    ) -> Result<()> {
        let leaf_page_num = path[path.len() - 1];

        // Get direct access to pages
        let pages = self.pages.get_mut();

        // Read and split the leaf
        let leaf_data = *pages
            .get(leaf_page_num)
            .ok_or_else(|| ZyronError::BTreeCorrupted("leaf not found".to_string()))?;
        let mut leaf = BTreeLeafPage::from_bytes(leaf_data);

        // Allocate new page
        let new_page_num = pages.allocate();
        let new_page_id = PageId::new(self.file_id, new_page_num);
        let (split_key, mut right_leaf) = leaf.split(new_page_id);

        // Insert into appropriate leaf
        if key.as_ref() < split_key.as_ref() {
            leaf.insert(key, tuple_id)?;
        } else {
            right_leaf.insert(key, tuple_id)?;
        }

        // Write both leaves
        pages.write(leaf_page_num, leaf.as_bytes());
        pages.write(new_page_num, right_leaf.as_bytes());

        // Propagate split up
        if path.len() < 2 {
            // Root was a leaf, create new root
            self.create_new_root_exclusive(split_key, new_page_num)
        } else {
            self.propagate_split_exclusive(split_key, new_page_num, path)
        }
    }

    /// Propagate split with exclusive access.
    fn propagate_split_exclusive(
        &mut self,
        key: Bytes,
        new_child: u32,
        path: &[u32],
    ) -> Result<()> {
        let mut current_key = key;
        let mut current_child = new_child;
        let mut parent_idx = path.len() - 2;

        loop {
            let parent_page_num = path[parent_idx];
            let pages = self.pages.get_mut();

            let parent_data = *pages
                .get(parent_page_num)
                .ok_or_else(|| ZyronError::BTreeCorrupted("parent not found".to_string()))?;
            let mut parent = BTreeInternalPage::from_bytes(parent_data);

            let new_child_page_id = PageId::new(self.file_id, current_child);

            match parent.insert(current_key.clone(), new_child_page_id) {
                Ok(()) => {
                    pages.write(parent_page_num, parent.as_bytes());
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Split the internal node
                    let new_page_num = pages.allocate();
                    let new_page_id = PageId::new(self.file_id, new_page_num);
                    let (promoted_key, mut right_internal) = parent.split(new_page_id);

                    if current_key.as_ref() < promoted_key.as_ref() {
                        parent.insert(current_key, new_child_page_id)?;
                    } else {
                        right_internal.insert(current_key, new_child_page_id)?;
                    }

                    pages.write(parent_page_num, parent.as_bytes());
                    pages.write(new_page_num, right_internal.as_bytes());

                    if parent_idx == 0 {
                        return self.create_new_root_exclusive(promoted_key, new_page_num);
                    }

                    current_key = promoted_key;
                    current_child = new_page_num;
                    parent_idx -= 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Create new root with exclusive access.
    fn create_new_root_exclusive(&mut self, key: Bytes, right_child: u32) -> Result<()> {
        let pages = self.pages.get_mut();
        let old_root = *self.root_page_num.get_mut();
        let height = *self.height.get_mut();

        let new_root_num = pages.allocate();
        let new_root_id = PageId::new(self.file_id, new_root_num);
        let old_root_id = PageId::new(self.file_id, old_root);
        let right_child_id = PageId::new(self.file_id, right_child);

        let mut new_root = BTreeInternalPage::new(new_root_id, height as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child_id)?;

        pages.write(new_root_num, new_root.as_bytes());

        *self.root_page_num.get_mut() = new_root_num;
        *self.height.get_mut() = height + 1;

        Ok(())
    }

    /// Searches with exclusive access (no locking).
    #[inline]
    pub fn search_exclusive(&mut self, key: &[u8]) -> Option<TupleId> {
        let pages = self.pages.get_mut();
        let height = *self.height.get_mut();
        let root = *self.root_page_num.get_mut();

        let leaf_page_num = Self::find_leaf_direct(pages, height, root, key);

        if let Some(data) = pages.get(leaf_page_num) {
            BTreeLeafPage::get_in_slice(data, key)
        } else {
            None
        }
    }

    /// Direct leaf lookup without any locking.
    #[inline]
    fn find_leaf_direct(pages: &InMemoryPageStore, height: u32, root: u32, key: &[u8]) -> u32 {
        let mut current = root;

        if height == 1 {
            return current;
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num;
            } else {
                break;
            }
        }

        current
    }

    /// Finds the path from root to leaf (returns page numbers).
    fn find_path_sync(&self, key: &[u8]) -> ([u32; Self::MAX_HEIGHT], usize) {
        let pages = self.pages.read();
        let height = self.height.load(std::sync::atomic::Ordering::Acquire);
        let root = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Acquire);

        let mut path = [0u32; Self::MAX_HEIGHT];
        let mut path_len = 0;
        let mut current = root;

        path[path_len] = current;
        path_len += 1;

        if height == 1 {
            return (path, path_len);
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num;
                path[path_len] = current;
                path_len += 1;
            } else {
                break;
            }
        }

        (path, path_len)
    }

    /// Insert with split handling (synchronous).
    fn insert_with_split_sync(&self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        let (path, path_len) = self.find_path_sync(&key);
        if path_len == 0 {
            return Err(ZyronError::BTreeCorrupted("empty path".to_string()));
        }

        let leaf_page_num = path[path_len - 1];

        let mut pages = self.pages.write();

        // Read the leaf page
        let leaf_data = pages
            .get(leaf_page_num)
            .ok_or_else(|| ZyronError::BTreeCorrupted("leaf not found".to_string()))?;
        let mut leaf = BTreeLeafPage::from_bytes(*leaf_data);

        // Allocate new page for right sibling
        let new_page_num = pages.allocate();
        let new_page_id = PageId::new(self.file_id, new_page_num);
        let (split_key, mut right_leaf) = leaf.split(new_page_id);

        // Insert the new key into appropriate leaf
        if key.as_ref() < split_key.as_ref() {
            leaf.insert(key, tuple_id)?;
        } else {
            right_leaf.insert(key, tuple_id)?;
        }

        // Write both leaves
        pages.write(leaf_page_num, leaf.as_bytes());
        pages.write(new_page_num, right_leaf.as_bytes());

        // Propagate split up the tree
        drop(pages); // Release lock before recursive call
        self.propagate_split_sync(split_key, new_page_num, &path[..path_len])
    }

    /// Propagate split up the tree.
    fn propagate_split_sync(&self, key: Bytes, new_child: u32, path: &[u32]) -> Result<()> {
        if path.len() < 2 {
            // Root was a leaf, create new root
            return self.create_new_root_sync(key, new_child);
        }

        let mut current_key = key;
        let mut current_child = new_child;
        let mut parent_idx = path.len() - 2;

        loop {
            let parent_page_num = path[parent_idx];

            let mut pages = self.pages.write();

            let parent_data = pages
                .get(parent_page_num)
                .ok_or_else(|| ZyronError::BTreeCorrupted("parent not found".to_string()))?;
            let mut parent = BTreeInternalPage::from_bytes(*parent_data);

            let new_child_page_id = PageId::new(self.file_id, current_child);

            match parent.insert(current_key.clone(), new_child_page_id) {
                Ok(()) => {
                    pages.write(parent_page_num, parent.as_bytes());
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Split the internal node
                    let new_page_num = pages.allocate();
                    let new_page_id = PageId::new(self.file_id, new_page_num);
                    let (promoted_key, mut right_internal) = parent.split(new_page_id);

                    // Insert into appropriate side
                    if current_key.as_ref() < promoted_key.as_ref() {
                        parent.insert(current_key, new_child_page_id)?;
                    } else {
                        right_internal.insert(current_key, new_child_page_id)?;
                    }

                    // Write both pages
                    pages.write(parent_page_num, parent.as_bytes());
                    pages.write(new_page_num, right_internal.as_bytes());

                    drop(pages);

                    // Continue propagating up
                    if parent_idx == 0 {
                        return self.create_new_root_sync(promoted_key, new_page_num);
                    }

                    current_key = promoted_key;
                    current_child = new_page_num;
                    parent_idx -= 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Creates a new root when the current root splits.
    fn create_new_root_sync(&self, key: Bytes, right_child: u32) -> Result<()> {
        let mut pages = self.pages.write();
        let old_root = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Acquire);
        let height = self.height.load(std::sync::atomic::Ordering::Acquire);

        let new_root_num = pages.allocate();
        let new_root_id = PageId::new(self.file_id, new_root_num);
        let old_root_id = PageId::new(self.file_id, old_root);
        let right_child_id = PageId::new(self.file_id, right_child);

        let mut new_root = BTreeInternalPage::new(new_root_id, height as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child_id)?;

        pages.write(new_root_num, new_root.as_bytes());

        self.root_page_num
            .store(new_root_num, std::sync::atomic::Ordering::Release);
        self.height
            .store(height + 1, std::sync::atomic::Ordering::Release);

        Ok(())
    }

    /// Deletes a key synchronously.
    pub fn delete_sync(&self, key: &[u8]) -> bool {
        let leaf_page_num = self.find_leaf_page_num(key);

        let mut pages = self.pages.write();
        if let Some(data) = pages.get(leaf_page_num) {
            let mut leaf = BTreeLeafPage::from_bytes(*data);
            match leaf.delete(key) {
                DeleteResult::Ok | DeleteResult::Underfull => {
                    pages.write(leaf_page_num, leaf.as_bytes());
                    true
                }
                DeleteResult::NotFound => false,
            }
        } else {
            false
        }
    }

    /// Range scan synchronously.
    pub fn range_scan_sync(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Vec<(Bytes, TupleId)> {
        let pages = self.pages.read();
        let mut results = Vec::new();

        // Find starting leaf
        let start_leaf_num = match start_key {
            Some(key) => self.find_leaf_page_num(key),
            None => self.find_leftmost_leaf_num(&pages),
        };

        let mut current_page_num = Some(start_leaf_num);

        while let Some(page_num) = current_page_num {
            if let Some(data) = pages.get(page_num) {
                let leaf = BTreeLeafPage::from_bytes(*data);

                for entry in leaf.entries() {
                    if let Some(start) = start_key {
                        if compare_keys(entry.key.as_ref(), start).is_lt() {
                            continue;
                        }
                    }

                    if let Some(end) = end_key {
                        if compare_keys(entry.key.as_ref(), end).is_gt() {
                            return results;
                        }
                    }

                    results.push((entry.key.clone(), entry.tuple_id));
                }

                current_page_num = leaf.next_leaf().map(|p| p.page_num);
            } else {
                break;
            }
        }

        results
    }

    /// Find leftmost leaf page number.
    fn find_leftmost_leaf_num(&self, pages: &InMemoryPageStore) -> u32 {
        let height = self.height.load(std::sync::atomic::Ordering::Acquire);
        let mut current = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Acquire);

        if height == 1 {
            return current;
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let internal = BTreeInternalPage::from_bytes(*data);
                current = internal.leftmost_child().page_num;
            } else {
                break;
            }
        }

        current
    }

    // =========================================================================
    // Async Wrappers (for API compatibility)
    // =========================================================================

    /// Searches for a key. Async wrapper around sync operation.
    pub async fn search(&self, key: &[u8]) -> Result<Option<TupleId>> {
        Ok(self.search_sync(key))
    }

    /// Inserts a key-value pair. Uses lock-free path since we have &mut self.
    pub async fn insert(&mut self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        self.insert_exclusive(key.as_ref(), tuple_id)
    }

    /// Deletes a key. Async wrapper around sync operation.
    pub async fn delete(&mut self, key: &[u8]) -> Result<bool> {
        Ok(self.delete_sync(key))
    }

    /// Range scan. Async wrapper around sync operation.
    pub async fn range_scan(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Result<Vec<(Bytes, TupleId)>> {
        Ok(self.range_scan_sync(start_key, end_key))
    }

    /// Scan all entries. Async wrapper around sync operation.
    pub async fn scan_all(&self) -> Result<Vec<(Bytes, TupleId)>> {
        Ok(self.range_scan_sync(None, None))
    }

    /// Flush is a no-op for in-memory B+Tree (persistence handled by checkpoint).
    pub async fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Batch insert multiple entries.
    pub async fn insert_batch(&mut self, entries: Vec<(Bytes, TupleId)>) -> Result<usize> {
        let mut inserted = 0;
        for (key, tuple_id) in entries {
            if key.len() <= MAX_KEY_SIZE {
                self.insert_sync(key.as_ref(), tuple_id)?;
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    /// Warm cache is a no-op for in-memory B+Tree (all data already in RAM).
    pub async fn warm_cache(&self) -> Result<usize> {
        Ok(0)
    }
}

//! Page-based B+Tree index implementation.

use super::checkpoint::{self, CheckpointConfig, CheckpointTrigger};
use super::constants::MAX_KEY_SIZE;
use super::page::{BTreeInternalPage, BTreeLeafPage};
use super::store::InMemoryPageStore;
use super::types::{DeleteResult, LeafPageHeader, compare_keys};
use crate::disk::DiskManager;
use crate::tuple::TupleId;
use bytes::Bytes;
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use zyron_buffer::BufferPool;
use zyron_common::page::PageId;
use zyron_common::{Result, ZyronError};

pub struct BTreeIndex {
    /// In-memory page storage (all nodes stored here).
    pages: RwLock<InMemoryPageStore>,
    /// Root page number (index into pages).
    root_page_num: AtomicU32,
    /// Tree height (1 = just root as leaf).
    height: AtomicU32,
    /// File ID for this index (used for PageId construction).
    file_id: u32,
    /// Disk manager for checkpoint I/O (not used in hot path).
    #[allow(dead_code)]
    disk: Arc<DiskManager>,
    /// Buffer pool reference (kept for compatibility, not used in hot path).
    #[allow(dead_code)]
    pool: Arc<BufferPool>,
    /// LSN of the most recent checkpoint. 0 means no checkpoint exists.
    checkpoint_lsn: AtomicU64,
    /// Directory for checkpoint files.
    checkpoint_dir: PathBuf,
    /// WAL bytes accumulated since last checkpoint. Lock-free counter
    /// so record_wal_bytes() on every WAL append has zero contention.
    wal_bytes_since_checkpoint: AtomicU64,
    /// WAL bytes threshold cached from CheckpointConfig. Allows
    /// maybe_checkpoint() to skip the Mutex when below threshold.
    wal_bytes_threshold: u64,
    /// Checkpoint config + last checkpoint time. Only locked when
    /// wal_bytes exceeds threshold (rare) and for reset after checkpoint.
    checkpoint_trigger: parking_lot::Mutex<CheckpointTrigger>,
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
        checkpoint_dir: PathBuf,
    ) -> Result<Self> {
        Self::create_with_config(
            disk,
            pool,
            file_id,
            checkpoint_dir,
            CheckpointConfig::default(),
        )
        .await
    }

    /// Creates a new B+ tree index with a custom checkpoint configuration.
    pub async fn create_with_config(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
        checkpoint_dir: PathBuf,
        checkpoint_config: CheckpointConfig,
    ) -> Result<Self> {
        // Create in-memory page store
        let mut store = InMemoryPageStore::new();

        // Allocate root leaf page (page 0)
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num as u64);

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
            checkpoint_lsn: AtomicU64::new(0),
            checkpoint_dir,
            wal_bytes_since_checkpoint: AtomicU64::new(0),
            wal_bytes_threshold: checkpoint_config.wal_bytes_threshold,
            checkpoint_trigger: parking_lot::Mutex::new(CheckpointTrigger::new(checkpoint_config)),
        })
    }

    /// Opens an existing B+ tree index, loading from checkpoint if available.
    ///
    /// If a .zyridx checkpoint file exists in checkpoint_dir, loads pages from it.
    /// On checkpoint load failure (corrupt file, bad CRC), falls back to creating
    /// an empty store. The caller is responsible for replaying WAL records after
    /// checkpoint_lsn to bring the index up to date.
    pub async fn open(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
        checkpoint_dir: &Path,
    ) -> Result<Self> {
        Self::open_with_config(
            disk,
            pool,
            file_id,
            checkpoint_dir,
            CheckpointConfig::default(),
        )
        .await
    }

    /// Opens an existing B+ tree index with a custom checkpoint configuration.
    pub async fn open_with_config(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
        checkpoint_dir: &Path,
        checkpoint_config: CheckpointConfig,
    ) -> Result<Self> {
        let checkpoint_path = checkpoint_dir.join(format!("index_{}.zyridx", file_id));

        // Try loading from checkpoint file directly into the page store.
        // V2 format: decompresses + bulk-builds the B+Tree. Returns height directly.
        if checkpoint_path.exists() {
            let mut store = InMemoryPageStore::new();
            match checkpoint::load_checkpoint_into_store(&checkpoint_path, &mut store, file_id) {
                Ok((lsn, root_page_num, _entry_count, height)) => {
                    return Ok(Self {
                        pages: RwLock::new(store),
                        root_page_num: AtomicU32::new(root_page_num),
                        height: AtomicU32::new(height),
                        file_id,
                        disk,
                        pool,
                        checkpoint_lsn: AtomicU64::new(lsn),
                        checkpoint_dir: checkpoint_dir.to_path_buf(),
                        wal_bytes_since_checkpoint: AtomicU64::new(0),
                        wal_bytes_threshold: checkpoint_config.wal_bytes_threshold,
                        checkpoint_trigger: parking_lot::Mutex::new(CheckpointTrigger::new(
                            checkpoint_config,
                        )),
                    });
                }
                Err(_) => {
                    // Corrupt checkpoint. Fall through to create empty store.
                    // Caller will do full WAL replay.
                }
            }
        }

        // No checkpoint or corrupt checkpoint. Create empty store.
        let mut store = InMemoryPageStore::new();
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num as u64);
        let root_page = BTreeLeafPage::new(root_page_id);
        store.write(root_page_num, root_page.as_bytes());

        Ok(Self {
            pages: RwLock::new(store),
            root_page_num: AtomicU32::new(root_page_num),
            height: AtomicU32::new(1),
            file_id,
            disk,
            pool,
            checkpoint_lsn: AtomicU64::new(0),
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            wal_bytes_since_checkpoint: AtomicU64::new(0),
            wal_bytes_threshold: checkpoint_config.wal_bytes_threshold,
            checkpoint_trigger: parking_lot::Mutex::new(CheckpointTrigger::new(checkpoint_config)),
        })
    }

    /// Returns the root page ID.
    #[inline]
    pub fn root_page_id(&self) -> PageId {
        PageId::new(
            self.file_id,
            self.root_page_num.load(Ordering::Acquire) as u64,
        )
    }

    /// Returns the file ID.
    #[inline]
    pub fn file_id(&self) -> u32 {
        self.file_id
    }

    /// Returns a read lock on the page store (for debugging/testing).
    #[cfg(test)]
    pub fn pages_ref(&self) -> parking_lot::RwLockReadGuard<'_, InMemoryPageStore> {
        self.pages.read()
    }

    /// Returns the tree height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height.load(Ordering::Acquire)
    }

    // =========================================================================
    // Core In-Memory Operations (Synchronous)
    // =========================================================================

    /// Finds the leaf page number for a given key using existing pages reference.
    #[inline]
    fn find_leaf_in_pages(&self, pages: &InMemoryPageStore, key: &[u8]) -> u32 {
        let height = self.height.load(Ordering::Relaxed);
        let mut current = self.root_page_num.load(Ordering::Relaxed);

        // Height 1 means root is the leaf
        if height == 1 {
            return current;
        }

        // Traverse internal nodes
        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num as u32;
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
    ///
    /// Uses optimistic lock-free reads via version stamps. Retries if a concurrent
    /// write is detected. Falls back to RwLock read on repeated contention.
    #[inline]
    pub fn search_sync(&self, key: &[u8]) -> Option<TupleId> {
        let pages = self.pages.read();

        // Optimistic path: traverse using try_read with version stamp validation.
        for _ in 0..4 {
            match self.search_optimistic(&pages, key) {
                Ok(result) => return result,
                Err(()) => {
                    // Version conflict, retry.
                    std::hint::spin_loop();
                }
            }
        }

        // Fallback: version-validated read after repeated contention (very rare).
        let leaf_page_num = self.find_leaf_in_pages(&pages, key);
        match pages.try_read(leaf_page_num) {
            Some(Ok(data)) => BTreeLeafPage::get_in_slice(&data, key),
            _ => None,
        }
    }

    /// Optimistic search using version stamps. Returns Err(()) if a version
    /// conflict is detected at any point during traversal.
    #[inline]
    fn search_optimistic(
        &self,
        pages: &InMemoryPageStore,
        key: &[u8],
    ) -> std::result::Result<Option<TupleId>, ()> {
        let height = self.height.load(Ordering::Acquire);
        let mut current = self.root_page_num.load(Ordering::Acquire);

        // Traverse internal nodes
        if height > 1 {
            for _ in 0..(height - 1) {
                let data = pages.try_read(current).ok_or(())??;
                let child_page_id = BTreeInternalPage::find_child_in_slice(&data, key);
                current = child_page_id.page_num as u32;
            }
        }

        // Read leaf page
        let leaf_data = pages.try_read(current).ok_or(())??;
        Ok(BTreeLeafPage::get_in_slice(&leaf_data, key))
    }

    /// Inserts a key-value pair synchronously. All data is in RAM.
    ///
    /// Uses optimistic lock-free path for non-split inserts: reads the leaf via
    /// version stamp, inserts into a local copy, CAS-writes back. Falls back to
    /// RwLock write on version conflict or when a split is needed.
    #[inline]
    pub fn insert_sync(&self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        if key.len() > MAX_KEY_SIZE {
            return Err(ZyronError::KeyTooLarge {
                size: key.len(),
                max: MAX_KEY_SIZE,
            });
        }

        // Optimistic fast path: try lock-free insert (no split case).
        let pages = self.pages.read();
        for _ in 0..4 {
            match self.insert_optimistic(&pages, key, tuple_id) {
                Ok(()) => return Ok(()),
                Err(ZyronError::NodeFull) => break, // Need split, fall through
                Err(ZyronError::VersionConflict) => {
                    std::hint::spin_loop();
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        drop(pages);

        // Locked path: handles version conflicts after retries and split case.
        {
            let mut pages = self.pages.write();
            let leaf_page_num = self.find_leaf_in_pages(&pages, key);

            if let Some(data) = pages.get_mut(leaf_page_num) {
                match BTreeLeafPage::insert_in_slice(data, key, tuple_id) {
                    Ok(()) => return Ok(()),
                    Err(ZyronError::NodeFull) => {
                        // Need split, fall through
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Slow path: need to split
        self.insert_with_split_sync(Bytes::copy_from_slice(key), tuple_id)
    }

    /// Optimistic lock-free insert. Reads the leaf page via version stamp,
    /// inserts into a local copy, CAS-writes the result back.
    /// Returns VersionConflict if a concurrent writer intervened.
    /// Returns NodeFull if the leaf needs a split.
    #[inline]
    fn insert_optimistic(
        &self,
        pages: &InMemoryPageStore,
        key: &[u8],
        tuple_id: TupleId,
    ) -> Result<()> {
        let height = self.height.load(Ordering::Acquire);
        let mut current = self.root_page_num.load(Ordering::Acquire);

        // Traverse internal nodes to find the leaf.
        if height > 1 {
            for _ in 0..(height - 1) {
                let data = match pages.try_read(current) {
                    Some(Ok(d)) => d,
                    _ => return Err(ZyronError::VersionConflict),
                };
                let child = BTreeInternalPage::find_child_in_slice(&data, key);
                current = child.page_num as u32;
            }
        }

        // Read the leaf page with its validated version in a single operation.
        let (mut leaf_data, version) = match pages.try_read_versioned(current) {
            Some(Ok(dv)) => dv,
            _ => return Err(ZyronError::VersionConflict),
        };

        // Insert into the local copy.
        BTreeLeafPage::insert_in_slice(&mut leaf_data, key, tuple_id)?;

        // CAS-write back. Fails if the version changed since our read.
        if pages.try_versioned_write(current, &leaf_data, version) {
            Ok(())
        } else {
            Err(ZyronError::VersionConflict)
        }
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
                    current = child.page_num as u32;
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
        let new_page_id = PageId::new(self.file_id, new_page_num as u64);
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

            let new_child_page_id = PageId::new(self.file_id, current_child as u64);

            match parent.insert(current_key.clone(), new_child_page_id) {
                Ok(()) => {
                    pages.write(parent_page_num, parent.as_bytes());
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Split the internal node
                    let new_page_num = pages.allocate();
                    let new_page_id = PageId::new(self.file_id, new_page_num as u64);
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
        let new_root_id = PageId::new(self.file_id, new_root_num as u64);
        let old_root_id = PageId::new(self.file_id, old_root as u64);
        let right_child_id = PageId::new(self.file_id, right_child as u64);

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
                current = child_page_id.page_num as u32;
            } else {
                break;
            }
        }

        current
    }

    /// Finds the path from root to leaf (returns page numbers).
    fn find_path_sync(&self, key: &[u8]) -> ([u32; Self::MAX_HEIGHT], usize) {
        let pages = self.pages.read();
        let height = self.height.load(Ordering::Acquire);
        let root = self.root_page_num.load(Ordering::Acquire);

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
                current = child_page_id.page_num as u32;
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
        let new_page_id = PageId::new(self.file_id, new_page_num as u64);
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

            let new_child_page_id = PageId::new(self.file_id, current_child as u64);

            match parent.insert(current_key.clone(), new_child_page_id) {
                Ok(()) => {
                    pages.write(parent_page_num, parent.as_bytes());
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Split the internal node
                    let new_page_num = pages.allocate();
                    let new_page_id = PageId::new(self.file_id, new_page_num as u64);
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
        let old_root = self.root_page_num.load(Ordering::Acquire);
        let height = self.height.load(Ordering::Acquire);

        let new_root_num = pages.allocate();
        let new_root_id = PageId::new(self.file_id, new_root_num as u64);
        let old_root_id = PageId::new(self.file_id, old_root as u64);
        let right_child_id = PageId::new(self.file_id, right_child as u64);

        let mut new_root = BTreeInternalPage::new(new_root_id, height as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child_id)?;

        pages.write(new_root_num, new_root.as_bytes());

        self.root_page_num.store(new_root_num, Ordering::Release);
        self.height.store(height + 1, Ordering::Release);

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

    /// Range scan synchronously. Works directly on borrowed page data
    /// without copying 16KB pages. Uses binary search to find the
    /// start position in the first leaf page.
    pub fn range_scan_sync(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Vec<(Bytes, TupleId)> {
        let pages = self.pages.read();
        let mut results = Vec::with_capacity(1024);

        let start_leaf_num = match start_key {
            Some(key) => self.find_leaf_page_num(key),
            None => self.find_leftmost_leaf_num(&pages),
        };

        let ho = LeafPageHeader::OFFSET;
        let sa = BTreeLeafPage::SLOT_ARRAY_START;
        let ss = BTreeLeafPage::SLOT_SIZE;
        let mut current_page_num = Some(start_leaf_num);
        let mut first_page = true;

        while let Some(pn) = current_page_num {
            let Some(data) = pages.get(pn) else { break };
            let ns = u16::from_le_bytes([data[ho], data[ho + 1]]) as usize;

            // Binary search for start position on first page
            let start_slot = if first_page {
                first_page = false;
                if let Some(sk) = start_key {
                    let mut lo = 0usize;
                    let mut hi = ns;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        let so = sa + mid * ss;
                        let eo = u16::from_le_bytes([data[so], data[so + 1]]) as usize;
                        let kl = u16::from_le_bytes([data[eo], data[eo + 1]]) as usize;
                        let ek = &data[eo + 2..eo + 2 + kl];
                        if compare_keys(ek, sk).is_lt() {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    lo
                } else {
                    0
                }
            } else {
                0
            };

            for slot_idx in start_slot..ns {
                let so = sa + slot_idx * ss;
                let eo = u16::from_le_bytes([data[so], data[so + 1]]) as usize;
                let kl = u16::from_le_bytes([data[eo], data[eo + 1]]) as usize;
                let ek = &data[eo + 2..eo + 2 + kl];

                if let Some(end) = end_key {
                    if compare_keys(ek, end).is_gt() {
                        return results;
                    }
                }

                let to = eo + 2 + kl;
                let pnv = u32::from_le_bytes([data[to], data[to + 1], data[to + 2], data[to + 3]]);
                let sid = u16::from_le_bytes([data[to + 4], data[to + 5]]);
                results.push((
                    Bytes::copy_from_slice(ek),
                    TupleId::new(PageId::new(0, pnv as u64), sid),
                ));
            }

            let next = u64::from_le_bytes([
                data[ho + 4],
                data[ho + 5],
                data[ho + 6],
                data[ho + 7],
                data[ho + 8],
                data[ho + 9],
                data[ho + 10],
                data[ho + 11],
            ]);
            current_page_num = if next == u64::MAX {
                None
            } else {
                Some(next as u32)
            };
        }

        results
    }

    /// Find leftmost leaf page number.
    fn find_leftmost_leaf_num(&self, pages: &InMemoryPageStore) -> u32 {
        let height = self.height.load(Ordering::Acquire);
        let mut current = self.root_page_num.load(Ordering::Acquire);

        if height == 1 {
            return current;
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let internal = BTreeInternalPage::from_bytes(*data);
                current = internal.leftmost_child().page_num as u32;
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

    /// Writes a compact V2 checkpoint of the current B+Tree to disk.
    ///
    /// Extracts all leaf entries, LZ4-compresses them, writes to a single file.
    /// Uses atomic rename (write to .tmp, rename to final) for crash safety.
    pub fn force_checkpoint(&self, current_lsn: u64) -> Result<()> {
        let pages = self.pages.read();
        let root = self.root_page_num.load(Ordering::Acquire);
        let height = self.height.load(Ordering::Acquire);
        let fsync = self.checkpoint_trigger.lock().config().fsync;

        let path = self
            .checkpoint_dir
            .join(format!("index_{}.zyridx", self.file_id));
        let tmp_path = self
            .checkpoint_dir
            .join(format!("index_{}.zyridx.tmp", self.file_id));

        std::fs::create_dir_all(&self.checkpoint_dir)?;
        checkpoint::write_checkpoint_from_store(
            &tmp_path,
            &pages,
            current_lsn,
            root,
            height,
            fsync,
        )?;
        std::fs::rename(&tmp_path, &path)?;

        self.checkpoint_lsn.store(current_lsn, Ordering::Release);
        Ok(())
    }

    /// Performs a graceful shutdown by writing a final checkpoint.
    ///
    /// Call this before dropping the BTreeIndex to persist all in-memory state.
    /// The caller provides the current WAL LSN so the next startup can skip
    /// replaying WAL records up to this point.
    pub fn shutdown(&self, current_lsn: u64) -> Result<()> {
        self.force_checkpoint(current_lsn)
    }

    /// Returns the LSN of the last completed checkpoint.
    #[inline]
    pub fn checkpoint_lsn(&self) -> u64 {
        self.checkpoint_lsn.load(Ordering::Acquire)
    }

    /// Flushes the index by writing a checkpoint at the given LSN.
    pub fn flush_with_lsn(&self, current_lsn: u64) -> Result<()> {
        self.force_checkpoint(current_lsn)
    }

    /// Records WAL bytes written since the last checkpoint.
    /// Lock-free: single fetch_add with Relaxed ordering.
    #[inline]
    pub fn record_wal_bytes(&self, bytes: u64) {
        self.wal_bytes_since_checkpoint
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Checks if a checkpoint should be triggered based on accumulated WAL
    /// bytes and elapsed time. If so, writes the checkpoint and resets the trigger.
    ///
    /// Fast path: single Relaxed atomic load. If below the WAL bytes threshold,
    /// returns immediately without acquiring the Mutex.
    pub fn maybe_checkpoint(&self, current_lsn: u64) -> Result<bool> {
        let wal_bytes = self.wal_bytes_since_checkpoint.load(Ordering::Relaxed);
        if wal_bytes < self.wal_bytes_threshold {
            return Ok(false);
        }
        let mut trigger = self.checkpoint_trigger.lock();
        if trigger.should_checkpoint(wal_bytes) {
            self.force_checkpoint(current_lsn)?;
            self.wal_bytes_since_checkpoint.store(0, Ordering::Relaxed);
            trigger.reset();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Flush is a no-op without a WAL LSN. Use flush_with_lsn() or
    /// force_checkpoint() for persistence.
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_insert_exclusive_10k() {
        let dir = tempdir().unwrap();
        let ckpt_dir = dir.path().join("ckpt");
        std::fs::create_dir_all(&ckpt_dir).unwrap();
        let disk = Arc::new(
            crate::DiskManager::new(crate::DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::auto_sized());
        let mut btree = BTreeIndex::create(disk, pool, 0, ckpt_dir).await.unwrap();
        for i in 0..10_000u64 {
            let key = i.to_be_bytes();
            let tid = TupleId::new(PageId::new(0, 0), 0);
            btree.insert_exclusive(&key, tid).unwrap();
        }
        assert!(btree.search_exclusive(&500u64.to_be_bytes()).is_some());
    }

    #[tokio::test]
    async fn test_insert_exclusive_1m_verify_all() {
        let dir = tempdir().unwrap();
        let ckpt_dir = dir.path().join("ckpt");
        std::fs::create_dir_all(&ckpt_dir).unwrap();
        let disk = Arc::new(
            crate::DiskManager::new(crate::DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::auto_sized());
        let mut btree = BTreeIndex::create(disk, pool, 0, ckpt_dir).await.unwrap();
        let n = 1_000_000u64;
        for i in 0..n {
            let key = i.to_be_bytes();
            let tid = TupleId::new(PageId::new(0, i % 1000), (i % 100) as u16);
            btree.insert_exclusive(&key, tid).unwrap();
        }
        eprintln!("Tree height: {}", btree.height());
        let mut missing = 0u64;
        let mut first_missing = None;
        for i in 0..n {
            let key = i.to_be_bytes();
            let expected = TupleId::new(PageId::new(0, i % 1000), (i % 100) as u16);
            let found = btree.search_exclusive(&key);
            if found != Some(expected) {
                missing += 1;
                if first_missing.is_none() {
                    first_missing = Some((i, found, expected));
                }
            }
        }
        if let Some((i, found, expected)) = first_missing {
            panic!(
                "First missing: {} (total: {}) found={:?} expected={:?}",
                i, missing, found, expected
            );
        }
    }
}

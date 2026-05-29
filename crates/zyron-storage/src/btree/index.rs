//! Page-based B+Tree index implementation.

use super::checkpoint::{self, CheckpointConfig, CheckpointTrigger};
use super::constants::MAX_KEY_SIZE;
use super::page::{BTreeInternalPage, BTreeLeafPage};
use super::store::InMemoryPageStore;
use super::types::{DeleteResult, LeafPageHeader, compare_keys};
use crate::tuple::TupleId;
use bytes::Bytes;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use zyron_common::page::PageId;
use zyron_common::{Result, ZyronError};

pub struct BTreeIndex {
    /// Lock-free in-memory page storage. See store.rs for the version
    /// protocol and lock_for_write semantics used by the methods below.
    pages: InMemoryPageStore,
    /// Root page number.
    root_page_num: AtomicU32,
    /// Tree height, 1 means root is also the only leaf.
    height: AtomicU32,
    /// File ID for this index, used for PageId construction.
    file_id: u32,
    /// LSN of the most recent checkpoint, 0 means no checkpoint exists.
    checkpoint_lsn: AtomicU64,
    /// Directory for checkpoint files.
    checkpoint_dir: PathBuf,
    /// WAL bytes accumulated since last checkpoint.
    wal_bytes_since_checkpoint: AtomicU64,
    /// WAL bytes threshold cached from CheckpointConfig.
    wal_bytes_threshold: u64,
    /// Checkpoint config + last checkpoint time, locked only when
    /// wal_bytes exceeds threshold and for reset after checkpoint.
    checkpoint_trigger: parking_lot::Mutex<CheckpointTrigger>,
    /// Serializes root-replacement during split propagation. Held only
    /// when the root itself splits, which is rare.
    root_change_lock: parking_lot::Mutex<()>,
}

impl BTreeIndex {
    /// Maximum B+Tree height (supports billions of keys).
    const MAX_HEIGHT: usize = 16;

    /// Creates a new B+ tree index. All pages live in RAM.
    pub async fn create(file_id: u32, checkpoint_dir: PathBuf) -> Result<Self> {
        Self::create_with_config(file_id, checkpoint_dir, CheckpointConfig::default()).await
    }

    /// Creates a new B+ tree index with a custom checkpoint configuration.
    pub async fn create_with_config(
        file_id: u32,
        checkpoint_dir: PathBuf,
        checkpoint_config: CheckpointConfig,
    ) -> Result<Self> {
        let mut store = InMemoryPageStore::new();
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num as u64);
        let root_page = BTreeLeafPage::new(root_page_id);
        store.write(root_page_num, root_page.as_bytes());

        Ok(Self {
            pages: store,
            root_page_num: AtomicU32::new(root_page_num),
            height: AtomicU32::new(1),
            file_id,
            checkpoint_lsn: AtomicU64::new(0),
            checkpoint_dir,
            wal_bytes_since_checkpoint: AtomicU64::new(0),
            wal_bytes_threshold: checkpoint_config.wal_bytes_threshold,
            checkpoint_trigger: parking_lot::Mutex::new(CheckpointTrigger::new(checkpoint_config)),
            root_change_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Opens an existing B+ tree index, loading from checkpoint if available.
    /// On checkpoint load failure (corrupt file, bad checksum), falls back to
    /// an empty store. The caller is responsible for replaying WAL records
    /// after checkpoint_lsn to bring the index up to date.
    pub async fn open(file_id: u32, checkpoint_dir: &Path) -> Result<Self> {
        Self::open_with_config(file_id, checkpoint_dir, CheckpointConfig::default()).await
    }

    /// Opens an existing B+ tree index with a custom checkpoint configuration.
    pub async fn open_with_config(
        file_id: u32,
        checkpoint_dir: &Path,
        checkpoint_config: CheckpointConfig,
    ) -> Result<Self> {
        let checkpoint_path = checkpoint_dir.join(format!("index_{}.zyridx", file_id));

        if checkpoint_path.exists() {
            let mut store = InMemoryPageStore::new();
            match checkpoint::load_checkpoint_into_store(&checkpoint_path, &mut store, file_id) {
                Ok((lsn, root_page_num, _entry_count, height)) => {
                    return Ok(Self {
                        pages: store,
                        root_page_num: AtomicU32::new(root_page_num),
                        height: AtomicU32::new(height),
                        file_id,
                        checkpoint_lsn: AtomicU64::new(lsn),
                        checkpoint_dir: checkpoint_dir.to_path_buf(),
                        wal_bytes_since_checkpoint: AtomicU64::new(0),
                        wal_bytes_threshold: checkpoint_config.wal_bytes_threshold,
                        checkpoint_trigger: parking_lot::Mutex::new(CheckpointTrigger::new(
                            checkpoint_config,
                        )),
                        root_change_lock: parking_lot::Mutex::new(()),
                    });
                }
                Err(_) => {
                    // Corrupt checkpoint, caller will do full WAL replay
                }
            }
        }

        let mut store = InMemoryPageStore::new();
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num as u64);
        let root_page = BTreeLeafPage::new(root_page_id);
        store.write(root_page_num, root_page.as_bytes());

        Ok(Self {
            pages: store,
            root_page_num: AtomicU32::new(root_page_num),
            height: AtomicU32::new(1),
            file_id,
            checkpoint_lsn: AtomicU64::new(0),
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            wal_bytes_since_checkpoint: AtomicU64::new(0),
            wal_bytes_threshold: checkpoint_config.wal_bytes_threshold,
            checkpoint_trigger: parking_lot::Mutex::new(CheckpointTrigger::new(checkpoint_config)),
            root_change_lock: parking_lot::Mutex::new(()),
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

    /// Returns a reference to the page store (for debugging/testing).
    #[cfg(test)]
    pub fn pages_ref(&self) -> &InMemoryPageStore {
        &self.pages
    }

    /// Returns the tree height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height.load(Ordering::Acquire)
    }

    // =========================================================================
    // Core In-Memory Operations (Synchronous)
    // =========================================================================

    /// Finds the leaf page number for a given key using lock-free
    /// version-stamped reads. Spins on torn reads.
    #[inline]
    fn find_leaf_in_pages(&self, pages: &InMemoryPageStore, key: &[u8]) -> u32 {
        loop {
            let height = self.height.load(Ordering::Acquire);
            let mut current = self.root_page_num.load(Ordering::Acquire);

            if height == 1 {
                return current;
            }

            let mut torn = false;
            for _ in 0..(height - 1) {
                match pages.try_read(current) {
                    Some(Ok(data)) => {
                        let child_page_id = BTreeInternalPage::find_child_in_slice(&data, key);
                        current = child_page_id.page_num as u32;
                    }
                    Some(Err(())) => {
                        torn = true;
                        break;
                    }
                    None => return current,
                }
            }
            if torn {
                std::hint::spin_loop();
                continue;
            }
            return current;
        }
    }

    /// Finds the leaf page number for a given key using lock-free reads.
    #[inline]
    fn find_leaf_page_num(&self, key: &[u8]) -> u32 {
        self.find_leaf_in_pages(&self.pages, key)
    }

    /// Searches for a key synchronously. All data is in RAM.
    ///
    /// Pure lock-free path: traverses via version-stamped reads, retrying
    /// on torn reads. No global lock acquisition.
    #[inline]
    pub fn search_sync(&self, key: &[u8]) -> Option<TupleId> {
        loop {
            match self.search_optimistic(&self.pages, key) {
                Ok(result) => return result,
                Err(()) => std::hint::spin_loop(),
            }
        }
    }

    /// Single optimistic-read attempt: `Ok` on a clean read, `Err(())` on a
    /// torn read (a concurrent writer was observed mid-update). Lets a caller
    /// run its own retry loop and count retries, e.g. a contention benchmark,
    /// without adding a counter to the production search_sync hot path.
    #[inline]
    pub fn search_attempt(&self, key: &[u8]) -> std::result::Result<Option<TupleId>, ()> {
        self.search_optimistic(&self.pages, key)
    }

    /// Optimistic search using version stamps. Returns Err(()) if a torn
    /// read is observed at any page in the traversal.
    ///
    /// After locating the candidate leaf, follows the right-link chain as
    /// long as the target exceeds the current leaf's last key. This is the
    /// Lehman-Yao protocol that handles the window between parent update
    /// and old-leaf shrink during a concurrent split: a reader that arrives
    /// at the pre-split leaf may find that its key was relocated to the
    /// right sibling, and follows the link to find it there.
    #[inline]
    fn search_optimistic(
        &self,
        pages: &InMemoryPageStore,
        key: &[u8],
    ) -> std::result::Result<Option<TupleId>, ()> {
        let height = self.height.load(Ordering::Acquire);
        let mut current = self.root_page_num.load(Ordering::Acquire);

        if height > 1 {
            for _ in 0..(height - 1) {
                let data = pages.try_read(current).ok_or(())??;
                let child_page_id = BTreeInternalPage::find_child_in_slice(&data, key);
                current = child_page_id.page_num as u32;
            }
        }

        let ho = LeafPageHeader::OFFSET;
        let sa = BTreeLeafPage::SLOT_ARRAY_START;
        let ss = BTreeLeafPage::SLOT_SIZE;
        // Cap right-link walks so a misbehaving leaf chain cannot stall
        // the reader. In practice we follow at most one link, the cap
        // covers transient mid-split states.
        for _ in 0..Self::MAX_HEIGHT * 4 {
            let leaf_data = pages.try_read(current).ok_or(())??;
            if let Some(tid) = BTreeLeafPage::get_in_slice(&leaf_data, key) {
                return Ok(Some(tid));
            }
            let ns = u16::from_le_bytes([leaf_data[ho], leaf_data[ho + 1]]) as usize;
            if ns == 0 {
                return Ok(None);
            }
            let last_so = sa + (ns - 1) * ss;
            let last_eo = u16::from_le_bytes([leaf_data[last_so], leaf_data[last_so + 1]]) as usize;
            let last_kl = u16::from_le_bytes([leaf_data[last_eo], leaf_data[last_eo + 1]]) as usize;
            let last_key = &leaf_data[last_eo + 2..last_eo + 2 + last_kl];
            if compare_keys(key, last_key).is_le() {
                return Ok(None);
            }
            let next_packed = u64::from_le_bytes([
                leaf_data[ho + 4],
                leaf_data[ho + 5],
                leaf_data[ho + 6],
                leaf_data[ho + 7],
                leaf_data[ho + 8],
                leaf_data[ho + 9],
                leaf_data[ho + 10],
                leaf_data[ho + 11],
            ]);
            if next_packed == u64::MAX {
                return Ok(None);
            }
            current = next_packed as u32;
        }
        Ok(None)
    }

    /// Inserts a key-value pair synchronously. All data is in RAM.
    ///
    /// Pure lock-free hot path: reads the leaf via version stamp, inserts
    /// into a local copy, CAS-writes back. Retries on version conflict.
    /// Acquires a per-leaf split latch only when an actual split is needed.
    #[inline]
    pub fn insert_sync(&self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        if key.len() > MAX_KEY_SIZE {
            return Err(ZyronError::KeyTooLarge {
                size: key.len(),
                max: MAX_KEY_SIZE,
            });
        }

        // Pure lock-free CAS loop. NodeFull breaks out to the split path,
        // VersionConflict spins. Under contention this still completes in
        // a few iterations because the CAS protocol guarantees forward
        // progress, one of the contending writers always wins per round.
        loop {
            match self.insert_optimistic(&self.pages, key, tuple_id) {
                Ok(()) => return Ok(()),
                Err(ZyronError::NodeFull) => break,
                Err(ZyronError::VersionConflict) => {
                    std::hint::spin_loop();
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Split path holds the per-leaf split latch, allocates a new page
        // via the lock-free counter, writes both halves via the version
        // protocol, and CAS-updates the parent to install the split key.
        self.insert_with_split_sync(Bytes::copy_from_slice(key), tuple_id)
    }

    /// Batched key insertion. Sorts the input by key, then locks each
    /// destination leaf once and inserts every key in the batch that
    /// belongs to that leaf, instead of doing 1 traversal + 1 lock per
    /// key as a loop of `insert_sync` calls would.
    ///
    /// For the seed workload (a 100-row INSERT VALUES batch) this collapses
    /// ~100 leaf round trips into ~1-2, which dominates the heap+WAL work.
    /// Items are passed `&mut` so the in-place sort is allocation-free.
    pub fn insert_many<K: AsRef<[u8]>>(&self, items: &mut [(K, TupleId)]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        for (key, _) in items.iter() {
            if key.as_ref().len() > MAX_KEY_SIZE {
                return Err(ZyronError::KeyTooLarge {
                    size: key.as_ref().len(),
                    max: MAX_KEY_SIZE,
                });
            }
        }
        items.sort_by(|a, b| compare_keys(a.0.as_ref(), b.0.as_ref()));

        let mut i = 0;
        while i < items.len() {
            // One lock-free descent per leaf group, returning the leaf and
            // its exclusive upper-bound key. The inner loop then routes the
            // sorted run with a cheap key comparison against the bound,
            // never re-descending (and never re-copying internal pages)
            // per key.
            let (leaf_pn, bound) = self.find_leaf_and_bound(items[i].0.as_ref());

            // Prepare-then-publish: build the updated leaf from a stable
            // version-stamped snapshot WITHOUT holding the page odd, then
            // publish the whole run with a single CAS. The page is odd only for
            // the microsecond-scale CAS+copy, not for the routing loop, so
            // concurrent readers see the old leaf intact and do not retry while
            // the run is being built. A publish conflict (another writer changed
            // the leaf between our read and CAS) rebuilds from a fresh snapshot.
            // This mirrors insert_optimistic; CAS interlocks with lock_for_write
            // splits via the same even/odd version, so the two stay correct.
            let mut applied = 0usize;
            let mut need_split = false;
            loop {
                let (data, version) = match self.pages.try_read_versioned(leaf_pn) {
                    Some(Ok(dv)) => dv,
                    // Writer active (odd) or torn read: re-read until stable.
                    Some(Err(())) => {
                        std::hint::spin_loop();
                        continue;
                    }
                    None => {
                        return Err(ZyronError::Internal(
                            "btree leaf page not installed during insert_many".into(),
                        ));
                    }
                };
                let mut leaf = BTreeLeafPage::from_bytes(data);

                let mut a = 0usize;
                let mut full = false;
                while i + a < items.len() {
                    // A key at or beyond the leaf's upper bound belongs in a
                    // sibling: stop and re-descend for it. Sorted order means
                    // every later key is also beyond the bound.
                    if let Some(ref b) = bound {
                        if compare_keys(items[i + a].0.as_ref(), b).is_ge() {
                            break;
                        }
                    }
                    let key = Bytes::copy_from_slice(items[i + a].0.as_ref());
                    match leaf.insert(key, items[i + a].1) {
                        Ok(()) => a += 1,
                        Err(ZyronError::NodeFull) => {
                            full = true;
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }

                if a == 0 {
                    // The first routed key did not fit (leaf full): nothing to
                    // publish, route it through the split path below.
                    need_split = full;
                    break;
                }

                // Publish the built leaf. On conflict, rebuild from a fresh read.
                if self
                    .pages
                    .try_versioned_write(leaf_pn, leaf.as_bytes(), version)
                {
                    applied = a;
                    need_split = full;
                    break;
                }
                std::hint::spin_loop();
            }

            i += applied;
            if need_split {
                // The current leaf cannot hold items[i], fall through to the
                // single-key split path which allocates a sibling, propagates
                // the split key into the parent, and shrinks the left half.
                let key = Bytes::copy_from_slice(items[i].0.as_ref());
                let tid = items[i].1;
                self.insert_with_split_sync(key, tid)?;
                i += 1;
            }
        }
        Ok(())
    }

    /// Finds the leaf for `key` plus that leaf's exclusive upper-bound key
    /// (the smallest separator strictly greater than every key the leaf can
    /// hold), via a single lock-free version-stamped descent. `None` bound
    /// means the leaf is the rightmost on its path and has no upper bound.
    fn find_leaf_and_bound(&self, key: &[u8]) -> (u32, Option<Vec<u8>>) {
        loop {
            let height = self.height.load(Ordering::Acquire);
            let mut current = self.root_page_num.load(Ordering::Acquire);
            if height == 1 {
                return (current, None);
            }

            let mut bound: Option<Vec<u8>> = None;
            let mut torn = false;
            for _ in 0..(height - 1) {
                match self.pages.try_read(current) {
                    Some(Ok(data)) => {
                        let (child, stop) = BTreeInternalPage::find_child_with_upper(&data, key);
                        current = child;
                        // The leaf's bound is the tightest (smallest)
                        // separator we stopped before across the path.
                        if let Some(s) = stop {
                            bound = Some(match bound {
                                Some(b) if compare_keys(&b, &s).is_le() => b,
                                _ => s,
                            });
                        }
                    }
                    Some(Err(())) => {
                        torn = true;
                        break;
                    }
                    None => return (current, bound),
                }
            }
            if torn {
                std::hint::spin_loop();
                continue;
            }
            return (current, bound);
        }
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

        let pages = &mut self.pages;
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
        let pages = &mut self.pages;

        // Read and split the leaf
        let leaf_data = *pages
            .get(leaf_page_num)
            .ok_or_else(|| ZyronError::BTreeCorrupted("leaf not found".to_string()))?;
        let mut leaf = BTreeLeafPage::from_bytes(leaf_data);

        // Allocate new page
        let new_page_num = pages.allocate();
        let new_page_id = PageId::new(self.file_id, new_page_num as u64);
        // Pass the inserting key so the split path can right-bias when the
        // key is the new rightmost, monotonic workloads (auto-increment,
        // UUID v7, time-ordered) avoid the 50% page-utilization penalty
        let (split_key, mut right_leaf) = leaf.split_for_key(Some(key.as_ref()), new_page_id);

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
            let pages = &mut self.pages;

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
        let pages = &mut self.pages;
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
        let pages = &mut self.pages;
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

    /// Finds the path from root to leaf using lock-free version-stamped
    /// reads. Spins on torn reads of any page along the descent.
    fn find_path_sync(&self, key: &[u8]) -> ([u32; Self::MAX_HEIGHT], usize) {
        loop {
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

            let mut torn = false;
            for _ in 0..(height - 1) {
                match self.pages.try_read(current) {
                    Some(Ok(data)) => {
                        let child_page_id = BTreeInternalPage::find_child_in_slice(&data, key);
                        current = child_page_id.page_num as u32;
                        path[path_len] = current;
                        path_len += 1;
                    }
                    Some(Err(())) => {
                        torn = true;
                        break;
                    }
                    None => return (path, path_len),
                }
            }
            if torn {
                std::hint::spin_loop();
                continue;
            }
            return (path, path_len);
        }
    }

    /// Insert with split handling, prepare-then-publish. The split halves are
    /// built off-lock from a version-stamped leaf snapshot and the right-half
    /// page is written off-lock (it is unreachable from the parent, so no
    /// reader observes it). The leaf is then held odd only for the publish:
    /// install the split key in the parent and write the shrunk left-half.
    /// Concurrent CAS writers and readers retry against the in-progress
    /// version only during that publish, not while the halves are built or
    /// the sibling is written. Unrelated leaves proceed in parallel because
    /// the lock is per-page.
    ///
    /// If the leaf changed between the snapshot read and the publish lock,
    /// the split is rebuilt from a fresh snapshot. The sibling page is
    /// allocated once and reused across rebuilds so a lost publish race does
    /// not leak pages.
    ///
    /// Publish ordering (leaf held odd):
    ///   1. Install the split key in the parent, making the right-half
    ///      reachable. Concurrent readers may now route to either half.
    ///   2. Write the shrunk left-half into the guard, which becomes
    ///      visible when the guard drops.
    fn insert_with_split_sync(&self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        let mut sibling: Option<(u32, PageId)> = None;

        loop {
            let (path, path_len) = self.find_path_sync(&key);
            if path_len == 0 {
                return Err(ZyronError::BTreeCorrupted("empty path".to_string()));
            }
            let leaf_page_num = path[path_len - 1];

            // Version-stamped snapshot read, no lock. Readers do not retry
            // while we build the split halves from this snapshot.
            let (leaf_bytes, version) = match self.pages.try_read_versioned(leaf_page_num) {
                Some(Ok(dv)) => dv,
                Some(Err(())) => {
                    std::hint::spin_loop();
                    continue;
                }
                None => {
                    return Err(ZyronError::BTreeCorrupted(
                        "leaf page not installed during split".to_string(),
                    ));
                }
            };
            let mut leaf = BTreeLeafPage::from_bytes(leaf_bytes);

            // The leaf may have gained room (a concurrent delete) since it
            // last read full. Publish a plain in-slice insert via CAS and
            // skip the structural split.
            match leaf.insert(key.clone(), tuple_id) {
                Ok(()) => {
                    if self
                        .pages
                        .try_versioned_write(leaf_page_num, leaf.as_bytes(), version)
                    {
                        return Ok(());
                    }
                    std::hint::spin_loop();
                    continue;
                }
                Err(ZyronError::NodeFull) => {}
                Err(e) => return Err(e),
            }

            // Allocate the sibling once, lazily, reused across rebuilds.
            let (sibling_pn, sibling_id) = *sibling.get_or_insert_with(|| {
                let pn = self.pages.allocate();
                (pn, PageId::new(self.file_id, pn as u64))
            });

            // Right-bias when the inserting key is the new rightmost so
            // monotonic workloads avoid the 50% page-utilization penalty
            // of midpoint splits.
            let (split_key, mut right_leaf) = leaf.split_for_key(Some(key.as_ref()), sibling_id);
            if key.as_ref() < split_key.as_ref() {
                leaf.insert(key.clone(), tuple_id)?;
            } else {
                right_leaf.insert(key.clone(), tuple_id)?;
            }

            // Write the right-half off-lock. Not reachable from the parent
            // until the publish below installs it, so no reader observes it.
            self.pages.force_write(sibling_pn, right_leaf.as_bytes());

            // Publish only if the leaf is byte-for-byte the snapshot we split.
            // A concurrent writer that changed it bumps the version, so we
            // rebuild from a fresh read.
            let leaf_guard = match self.pages.try_lock_for_write_at(leaf_page_num, version) {
                Some(g) => g,
                None => {
                    std::hint::spin_loop();
                    continue;
                }
            };

            // Step 1: install split key in parent (makes the right-half
            // reachable). Step 2: shrink left-half, visible when guard drops.
            self.propagate_split_sync(split_key, sibling_pn, &path[..path_len])?;
            leaf_guard.write(leaf.as_bytes());
            return Ok(());
        }
    }

    /// Propagate a split up the tree. Each parent is locked individually
    /// via lock_for_write so unrelated splits at sibling subtrees proceed
    /// in parallel. The lock is held across the parent read+mutate so
    /// concurrent CAS writers on the parent (none today, but the protocol
    /// is symmetric) would retry against the in-progress version.
    fn propagate_split_sync(&self, key: Bytes, new_child: u32, path: &[u32]) -> Result<()> {
        if path.len() < 2 {
            return self.create_new_root_sync(key, new_child);
        }

        let mut current_key = key;
        let mut current_child = new_child;
        let mut parent_idx = path.len() - 2;

        loop {
            let parent_page_num = path[parent_idx];
            let parent_guard = self.pages.lock_for_write(parent_page_num);

            let parent_bytes = parent_guard.read();
            let mut parent = BTreeInternalPage::from_bytes(parent_bytes);

            let new_child_page_id = PageId::new(self.file_id, current_child as u64);

            match parent.insert(current_key.clone(), new_child_page_id) {
                Ok(()) => {
                    parent_guard.write(parent.as_bytes());
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Internal-node split. Build the right half on a fresh
                    // page and publish it before mutating the left half so
                    // readers see a consistent routing graph throughout.
                    let new_page_num = self.pages.allocate();
                    let new_page_id = PageId::new(self.file_id, new_page_num as u64);
                    let (promoted_key, mut right_internal) = parent.split(new_page_id);

                    if current_key.as_ref() < promoted_key.as_ref() {
                        parent.insert(current_key, new_child_page_id)?;
                    } else {
                        right_internal.insert(current_key, new_child_page_id)?;
                    }

                    self.pages
                        .force_write(new_page_num, right_internal.as_bytes());

                    if parent_idx == 0 {
                        self.create_new_root_sync(promoted_key.clone(), new_page_num)?;
                        parent_guard.write(parent.as_bytes());
                        return Ok(());
                    }

                    parent_guard.write(parent.as_bytes());
                    drop(parent_guard);

                    current_key = promoted_key;
                    current_child = new_page_num;
                    parent_idx -= 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Creates a new root when the current root splits. The root_change_lock
    /// guarantees that concurrent root splits are linearized, which is
    /// trivially cheap because the root only splits once per tree-height
    /// increase.
    fn create_new_root_sync(&self, key: Bytes, right_child: u32) -> Result<()> {
        let _g = self.root_change_lock.lock();
        let old_root = self.root_page_num.load(Ordering::Acquire);
        let height = self.height.load(Ordering::Acquire);

        let new_root_num = self.pages.allocate();
        let new_root_id = PageId::new(self.file_id, new_root_num as u64);
        let old_root_id = PageId::new(self.file_id, old_root as u64);
        let right_child_id = PageId::new(self.file_id, right_child as u64);

        let mut new_root = BTreeInternalPage::new(new_root_id, height as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child_id)?;

        self.pages.force_write(new_root_num, new_root.as_bytes());

        self.root_page_num.store(new_root_num, Ordering::Release);
        self.height.store(height + 1, Ordering::Release);

        Ok(())
    }

    /// Deletes a key synchronously. The leaf is locked for the rewrite
    /// so concurrent inserters and other deleters retry against the
    /// in-progress version, and readers see the deletion atomically when
    /// the guard drops.
    pub fn delete_sync(&self, key: &[u8]) -> bool {
        let leaf_page_num = self.find_leaf_page_num(key);
        let guard = self.pages.lock_for_write(leaf_page_num);
        let mut leaf = BTreeLeafPage::from_bytes(guard.read());
        match leaf.delete(key) {
            DeleteResult::Ok | DeleteResult::Underfull => {
                guard.write(leaf.as_bytes());
                true
            }
            DeleteResult::NotFound => false,
        }
    }

    /// Range scan synchronously. Each leaf page is read via the stable
    /// version protocol, producing a per-page 8KB copy on the stack. This
    /// removes the global read lock the previous implementation required
    /// to hold across the entire scan.
    pub fn range_scan_sync(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Vec<(Bytes, TupleId)> {
        let mut results = Vec::with_capacity(1024);
        let start_leaf_num = match start_key {
            Some(key) => self.find_leaf_in_pages(&self.pages, key),
            None => self.find_leftmost_leaf_num(&self.pages),
        };

        let ho = LeafPageHeader::OFFSET;
        let sa = BTreeLeafPage::SLOT_ARRAY_START;
        let ss = BTreeLeafPage::SLOT_SIZE;
        let mut current_page_num = Some(start_leaf_num);
        let mut first_page = true;

        while let Some(pn) = current_page_num {
            let Some(data) = self.pages.read_stable(pn) else {
                break;
            };
            let data = &data;
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

                if let Some(end) = end_key
                    && compare_keys(ek, end).is_gt()
                {
                    return results;
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

    /// Range scan that calls a callback for each matching entry. Each leaf
    /// page is materialized once via read_stable, so the callback sees a
    /// consistent snapshot of that leaf even with concurrent splits.
    pub fn range_scan_for_each<F>(&self, start_key: Option<&[u8]>, end_key: Option<&[u8]>, mut f: F)
    where
        F: FnMut(&[u8], TupleId) -> bool,
    {
        let start_leaf_num = match start_key {
            Some(key) => self.find_leaf_in_pages(&self.pages, key),
            None => self.find_leftmost_leaf_num(&self.pages),
        };

        let ho = LeafPageHeader::OFFSET;
        let sa = BTreeLeafPage::SLOT_ARRAY_START;
        let ss = BTreeLeafPage::SLOT_SIZE;
        let mut current_page_num = Some(start_leaf_num);
        let mut first_page = true;

        while let Some(pn) = current_page_num {
            let Some(data) = self.pages.read_stable(pn) else {
                break;
            };
            let data = &data;
            let ns = u16::from_le_bytes([data[ho], data[ho + 1]]) as usize;

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

                if let Some(end) = end_key
                    && compare_keys(ek, end).is_gt()
                {
                    return;
                }

                let to = eo + 2 + kl;
                let pnv = u32::from_le_bytes([data[to], data[to + 1], data[to + 2], data[to + 3]]);
                let sid = u16::from_le_bytes([data[to + 4], data[to + 5]]);
                if !f(ek, TupleId::new(PageId::new(0, pnv as u64), sid)) {
                    return;
                }
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
    }

    /// Find leftmost leaf page number using lock-free reads.
    fn find_leftmost_leaf_num(&self, pages: &InMemoryPageStore) -> u32 {
        loop {
            let height = self.height.load(Ordering::Acquire);
            let mut current = self.root_page_num.load(Ordering::Acquire);

            if height == 1 {
                return current;
            }

            let mut torn = false;
            for _ in 0..(height - 1) {
                match pages.try_read(current) {
                    Some(Ok(data)) => {
                        let internal = BTreeInternalPage::from_bytes(data);
                        current = internal.leftmost_child().page_num as u32;
                    }
                    Some(Err(())) => {
                        torn = true;
                        break;
                    }
                    None => return current,
                }
            }
            if torn {
                std::hint::spin_loop();
                continue;
            }
            return current;
        }
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
        // Hold root_change_lock so the tree height and root_page_num
        // do not shift mid-checkpoint. Leaf-level splits below the root
        // still proceed, the checkpoint LSN ensures any post-snapshot
        // structural changes get replayed on next recovery.
        let _g = self.root_change_lock.lock();
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
            &self.pages,
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
        let mut btree = BTreeIndex::create(0, ckpt_dir).await.unwrap();
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
        let mut btree = BTreeIndex::create(0, ckpt_dir).await.unwrap();
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

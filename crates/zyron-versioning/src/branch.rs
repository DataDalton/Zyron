//! Git-like table branching with copy-on-write page management.
//!
//! Branches share pages with their parent until modified. The BranchManager
//! uses an atomic bitset for fast "is modified?" checks before falling
//! through to the hash map for actual page resolution.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::error::{Result, ZyronError};
use zyron_common::page::PageId;

use crate::version::VersionId;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Unique identifier for a branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BranchId(pub u64);

impl std::fmt::Display for BranchId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "branch_{}", self.0)
    }
}

/// Metadata for a branch.
#[derive(Debug, Clone)]
pub struct BranchEntry {
    pub id: BranchId,
    pub name: String,
    pub parent_branch_id: Option<BranchId>,
    /// Version at the time the branch was created.
    pub base_version_id: VersionId,
    /// Creation timestamp in microseconds since epoch.
    pub created_at: i64,
    pub description: String,
    pub is_active: bool,
}

/// Result of merging one branch into another.
#[derive(Debug)]
pub struct MergeResult {
    /// Number of pages successfully merged.
    pub merged_pages: u64,
    /// Conflicting pages (both branches modified the same page).
    pub conflicts: Vec<ConflictEntry>,
    /// New version created by the merge (if successful).
    pub result_version: VersionId,
}

/// A single page conflict detected during merge.
#[derive(Debug, Clone)]
pub struct ConflictEntry {
    pub page_id: PageId,
    pub conflict_type: ConflictType,
}

/// Classification of a merge conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Both source and target branches modified the same page.
    BothModified,
    /// Page was deleted in target but modified in source.
    DeletedInTarget,
    /// Page was deleted in source but modified in target.
    DeletedInSource,
}

// ---------------------------------------------------------------------------
// Page tracker with bitset fast path
// ---------------------------------------------------------------------------

/// Tracks which pages a branch has modified locally.
///
/// Uses an atomic bitset for fast "is modified?" queries. If the bit is
/// not set, the page is guaranteed unmodified (no hash map lookup needed).
/// If the bit is set, the hash map is consulted for the actual override.
struct BranchPageTracker {
    /// Atomic bitset. 1 bit per page number (mod 64 indexing per AtomicU64).
    modified_bitset: Vec<AtomicU64>,
    /// Actual page overrides: original page_id -> branch-local page_id.
    overrides: scc::HashMap<PageId, PageId>,
}

impl BranchPageTracker {
    /// Creates a new tracker with capacity for the given number of pages.
    fn new(initial_page_capacity: u64) -> Self {
        let words = ((initial_page_capacity + 63) / 64) as usize;
        let words = words.max(16); // minimum 16 words = 1024 pages
        let mut bitset = Vec::with_capacity(words);
        for _ in 0..words {
            bitset.push(AtomicU64::new(0));
        }
        Self {
            modified_bitset: bitset,
            overrides: scc::HashMap::new(),
        }
    }

    /// Returns true if the page is definitely not modified (bitset fast path).
    /// Returns false if the page might be modified (needs HashMap check).
    #[inline]
    fn is_definitely_unmodified(&self, page_num: u64) -> bool {
        let word_idx = (page_num / 64) as usize;
        if word_idx >= self.modified_bitset.len() {
            // Beyond bitset capacity, cannot rule out modification.
            // Fall through to HashMap check.
            return false;
        }
        let bit = 1u64 << (page_num % 64);
        self.modified_bitset[word_idx].load(Ordering::Relaxed) & bit == 0
    }

    /// Records a page modification.
    fn set_modified(&self, page_num: u64, original: PageId, local: PageId) {
        self.mark_bit(page_num);
        // Always insert into HashMap regardless of bitset capacity
        let _ = self.overrides.insert_sync(original, local);
    }

    /// Sets the bitset bit for a page number.
    #[inline]
    fn mark_bit(&self, page_num: u64) {
        let word_idx = (page_num / 64) as usize;
        if word_idx < self.modified_bitset.len() {
            let bit = 1u64 << (page_num % 64);
            self.modified_bitset[word_idx].fetch_or(bit, Ordering::Relaxed);
        }
    }

    /// Inserts an override only when the original page has none yet, returning
    /// the page that wins. Used by copy-on-write so two writers copying the same
    /// page converge on one branch-local page.
    fn get_or_insert_override(&self, original: PageId, local: PageId) -> PageId {
        match self.overrides.insert_sync(original, local) {
            Ok(()) => {
                self.mark_bit(original.page_num);
                local
            }
            Err(_) => self
                .overrides
                .read_sync(&original, |_, l| *l)
                .unwrap_or(local),
        }
    }

    /// Resolves a page: returns the branch-local page_id if modified.
    fn resolve(&self, page_id: PageId) -> Option<PageId> {
        if self.is_definitely_unmodified(page_id.page_num) {
            return None;
        }
        self.overrides.read_sync(&page_id, |_, local| *local)
    }

    /// Returns the number of modified pages.
    fn modified_count(&self) -> usize {
        self.overrides.len()
    }

    /// Iterates over all overrides.
    fn for_each_override(&self, mut f: impl FnMut(PageId, PageId)) {
        self.overrides.iter_sync(|original, local| {
            f(*original, *local);
            true
        });
    }
}

// ---------------------------------------------------------------------------
// BranchManager
// ---------------------------------------------------------------------------

/// Manages table branches with copy-on-write page semantics.
///
/// Each branch tracks page overrides. Reads resolve through the branch
/// chain (child -> parent) until an override or the original page is found.
pub struct BranchManager {
    branches: scc::HashMap<u64, BranchEntry>,
    page_trackers: scc::HashMap<u64, BranchPageTracker>,
    /// Branch-local overlay file ids per (branch_id, table heap_file_id).
    branch_table_files: scc::HashMap<u128, zyron_common::BranchFiles>,
    /// Count of pages a branch appended per (branch_id, table heap_file_id).
    append_counts: scc::HashMap<u128, u64>,
    next_branch_id: AtomicU64,
    /// Allocator for branch-local overlay file ids. Starts above user table and
    /// index file id ranges so branch files never collide.
    next_branch_file_id: std::sync::atomic::AtomicU32,
    data_dir: PathBuf,
}

/// First file id handed out for branch-local overlay files. User table heap
/// files start at 200 and index files at 10000, so this range is disjoint.
const BRANCH_FILE_ID_BASE: u32 = 2_000_000;

/// Packs a (branch_id, heap_file_id) pair into one key.
#[inline]
fn branch_table_key(branch_id: u64, heap_file_id: u32) -> u128 {
    ((branch_id as u128) << 32) | (heap_file_id as u128)
}

impl BranchManager {
    /// Creates a new branch manager.
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            branches: scc::HashMap::new(),
            page_trackers: scc::HashMap::new(),
            branch_table_files: scc::HashMap::new(),
            append_counts: scc::HashMap::new(),
            next_branch_id: AtomicU64::new(1),
            next_branch_file_id: std::sync::atomic::AtomicU32::new(BRANCH_FILE_ID_BASE),
            data_dir,
        }
    }

    /// Creates a new branch.
    pub fn create_branch(
        &self,
        name: &str,
        parent: Option<BranchId>,
        base_version: VersionId,
        description: &str,
        now_micros: i64,
    ) -> Result<BranchId> {
        // Check name uniqueness
        let mut name_exists = false;
        self.branches.iter_sync(|_, entry| {
            if entry.is_active && entry.name == name {
                name_exists = true;
            }
            true
        });
        if name_exists {
            return Err(ZyronError::BranchAlreadyExists(name.to_string()));
        }

        let id = BranchId(self.next_branch_id.fetch_add(1, Ordering::Relaxed));

        let entry = BranchEntry {
            id,
            name: name.to_string(),
            parent_branch_id: parent,
            base_version_id: base_version,
            created_at: now_micros,
            description: description.to_string(),
            is_active: true,
        };

        let _ = self.branches.insert_sync(id.0, entry);
        let _ = self
            .page_trackers
            .insert_sync(id.0, BranchPageTracker::new(1024));

        // Persistence is the caller's responsibility (the DDL command boundary)
        // so the in-memory create stays off the fsync path.
        Ok(id)
    }

    /// Resolves a page for a branch, returning the actual page to read.
    ///
    /// Checks the branch's override map first (bitset fast path), then
    /// walks up the parent chain iteratively. Returns the original page_id
    /// if no overrides exist in the branch chain.
    pub fn resolve_page(&self, branch_id: BranchId, page_id: PageId) -> PageId {
        let mut current = branch_id;
        loop {
            // Check this branch's tracker
            let resolved = self
                .page_trackers
                .read_sync(&current.0, |_, tracker| tracker.resolve(page_id));

            if let Some(Some(local_page)) = resolved {
                return local_page;
            }

            // Walk to parent branch
            let parent = self
                .branches
                .read_sync(&current.0, |_, entry| entry.parent_branch_id);

            match parent {
                Some(Some(parent_id)) => current = parent_id,
                _ => return page_id,
            }
        }
    }

    /// Records a copy-on-write page modification for a branch.
    ///
    /// The caller is responsible for actually copying the page data and
    /// allocating the local_page_id through the DiskManager.
    pub fn record_page_override(
        &self,
        branch_id: BranchId,
        original_page_id: PageId,
        local_page_id: PageId,
    ) -> Result<()> {
        let found = self.page_trackers.read_sync(&branch_id.0, |_, tracker| {
            tracker.set_modified(original_page_id.page_num, original_page_id, local_page_id);
        });
        if found.is_none() {
            return Err(ZyronError::BranchNotFound(format!("{}", branch_id)));
        }
        Ok(())
    }

    /// Returns the file_id base for a branch's local pages.
    ///
    /// Branch pages use file_id range: 50000 + branch_id * 1000.
    pub fn branch_file_id_base(branch_id: BranchId) -> u32 {
        50000 + (branch_id.0 as u32) * 1000
    }

    /// Merges source branch into target branch.
    ///
    /// For each page overridden in the source:
    /// - If not overridden in target: copy the override to target
    /// - If also overridden in target: record a conflict
    pub fn merge_branch(
        &self,
        source: BranchId,
        target: BranchId,
        result_version: VersionId,
    ) -> Result<MergeResult> {
        let source_exists = self.branches.contains_sync(&source.0);
        if !source_exists {
            return Err(ZyronError::BranchNotFound(format!("{}", source)));
        }
        let target_exists = self.branches.contains_sync(&target.0);
        if !target_exists {
            return Err(ZyronError::BranchNotFound(format!("{}", target)));
        }

        let mut conflicts = Vec::new();
        let mut merged_pages = 0u64;

        // Collect source overrides
        let mut source_overrides = Vec::new();
        self.page_trackers.read_sync(&source.0, |_, tracker| {
            tracker.for_each_override(|original, local| {
                source_overrides.push((original, local));
            });
        });

        // Apply each source override to target
        for (original_page, source_local) in &source_overrides {
            let target_has_override = self.page_trackers.read_sync(&target.0, |_, tracker| {
                tracker.resolve(*original_page).is_some()
            });

            match target_has_override {
                Some(true) => {
                    // Conflict: both branches modified the same page
                    conflicts.push(ConflictEntry {
                        page_id: *original_page,
                        conflict_type: ConflictType::BothModified,
                    });
                }
                _ => {
                    // No conflict: apply source override to target
                    self.page_trackers.read_sync(&target.0, |_, tracker| {
                        tracker.set_modified(original_page.page_num, *original_page, *source_local);
                    });
                    merged_pages += 1;
                }
            }
        }

        Ok(MergeResult {
            merged_pages,
            conflicts,
            result_version,
        })
    }

    /// Deletes a branch and its page overrides, removes its copy-on-write
    /// overlay records (file-id map and append counts), and reclaims the cow,
    /// append, and append-fsm data files it allocated. Without this the overlay
    /// records survive restart pointing at a branch that no longer exists and
    /// the overlay data files leak.
    pub fn delete_branch(&self, branch_id: BranchId) -> Result<()> {
        let removed = self.branches.update_sync(&branch_id.0, |_, entry| {
            entry.is_active = false;
        });
        if removed.is_none() {
            return Err(ZyronError::BranchNotFound(format!("{}", branch_id)));
        }
        let _ = self.page_trackers.remove_sync(&branch_id.0);

        // Collect the branch's overlay keys and the data files they allocated.
        let mut keys: Vec<u128> = Vec::new();
        let mut files: Vec<zyron_common::BranchFiles> = Vec::new();
        self.branch_table_files.iter_sync(|k, f| {
            if (*k >> 32) as u64 == branch_id.0 {
                keys.push(*k);
                files.push(*f);
            }
            true
        });
        for k in &keys {
            let _ = self.branch_table_files.remove_sync(k);
        }

        let mut count_keys: Vec<u128> = Vec::new();
        self.append_counts.iter_sync(|k, _| {
            if (*k >> 32) as u64 == branch_id.0 {
                count_keys.push(*k);
            }
            true
        });
        for k in &count_keys {
            let _ = self.append_counts.remove_sync(k);
        }

        // Reclaim the overlay data files. A missing file is not an error since
        // a branch that never wrote to a table has no file on disk.
        for f in &files {
            for file_id in [f.cow_file_id, f.append_file_id, f.append_fsm_file_id] {
                let path = self.data_dir.join(format!("{:08}.dat", file_id));
                match std::fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(ZyronError::Io(e)),
                }
            }
        }
        Ok(())
    }

    /// Returns all active branches.
    pub fn list_branches(&self) -> Vec<BranchEntry> {
        let mut result = Vec::new();
        self.branches.iter_sync(|_, entry| {
            if entry.is_active {
                result.push(entry.clone());
            }
            true
        });
        result
    }

    /// Finds a branch by name.
    pub fn get_branch_by_name(&self, name: &str) -> Result<BranchEntry> {
        let mut found: Option<BranchEntry> = None;
        self.branches.iter_sync(|_, entry| {
            if entry.is_active && entry.name == name {
                found = Some(entry.clone());
            }
            true
        });
        found.ok_or_else(|| ZyronError::BranchNotFound(name.to_string()))
    }

    /// Returns a branch entry by ID.
    pub fn get_branch(&self, id: BranchId) -> Result<BranchEntry> {
        self.branches
            .read_sync(&id.0, |_, entry| entry.clone())
            .ok_or_else(|| ZyronError::BranchNotFound(format!("{}", id)))
    }

    /// Returns the number of modified pages in a branch.
    pub fn modified_page_count(&self, branch_id: BranchId) -> usize {
        self.page_trackers
            .read_sync(&branch_id.0, |_, tracker| tracker.modified_count())
            .unwrap_or(0)
    }

    /// Returns the branch's copy-on-write overrides for one table as
    /// (original main page, branch cow page) pairs. Used by MERGE to apply the
    /// branch's row tombstones to the main line.
    pub fn cow_overrides(&self, branch_id: BranchId, heap_file_id: u32) -> Vec<(PageId, PageId)> {
        let mut out = Vec::new();
        self.page_trackers.read_sync(&branch_id.0, |_, tracker| {
            tracker.for_each_override(|original, local| {
                if original.file_id == heap_file_id {
                    out.push((original, local));
                }
            });
        });
        out
    }

    /// Returns a branch's overlay file ids for a table without allocating them.
    /// None when the branch never wrote to the table.
    pub fn branch_files_lookup(
        &self,
        branch_id: BranchId,
        heap_file_id: u32,
    ) -> Option<zyron_common::BranchFiles> {
        self.branch_table_files
            .read_sync(&branch_table_key(branch_id.0, heap_file_id), |_, f| *f)
    }

    /// Returns the number of pages a branch appended for a table.
    pub fn append_pages(&self, branch_id: BranchId, heap_file_id: u32) -> u64 {
        self.append_counts
            .read_sync(&branch_table_key(branch_id.0, heap_file_id), |_, v| *v)
            .unwrap_or(0)
    }

    /// Path of the branch metadata file.
    fn state_path(&self) -> PathBuf {
        self.data_dir.join(".zybranches")
    }

    /// Persists branch metadata and the copy-on-write overlay (file-id map,
    /// append page counts, page overrides, and the file-id allocator) to disk
    /// with an atomic rename so a crash mid-write never leaves a torn file. The
    /// overlay page bytes themselves live in their own data files and are
    /// flushed through the buffer pool; this records the metadata needed to
    /// resolve them again after restart.
    pub fn persist(&self) -> Result<()> {
        let branches = self.list_branches();
        let mut buf = Vec::with_capacity(128 + branches.len() * 64);
        buf.extend_from_slice(&(branches.len() as u32).to_le_bytes());
        for b in &branches {
            buf.extend_from_slice(&b.id.0.to_le_bytes());
            match b.parent_branch_id {
                Some(p) => {
                    buf.push(1);
                    buf.extend_from_slice(&p.0.to_le_bytes());
                }
                None => buf.push(0),
            }
            buf.extend_from_slice(&b.base_version_id.0.to_le_bytes());
            buf.extend_from_slice(&b.created_at.to_le_bytes());
            write_str(&mut buf, &b.name);
            write_str(&mut buf, &b.description);
        }

        // File-id allocator high-water mark so reused ids never collide.
        buf.extend_from_slice(
            &self
                .next_branch_file_id
                .load(Ordering::Relaxed)
                .to_le_bytes(),
        );

        // Branch overlay file ids per (branch, table).
        let mut files_entries: Vec<(u64, u32, zyron_common::BranchFiles)> = Vec::new();
        self.branch_table_files.iter_sync(|k, f| {
            files_entries.push(((*k >> 32) as u64, (*k & 0xFFFF_FFFF) as u32, *f));
            true
        });
        buf.extend_from_slice(&(files_entries.len() as u32).to_le_bytes());
        for (bid, hfid, f) in &files_entries {
            buf.extend_from_slice(&bid.to_le_bytes());
            buf.extend_from_slice(&hfid.to_le_bytes());
            buf.extend_from_slice(&f.cow_file_id.to_le_bytes());
            buf.extend_from_slice(&f.append_file_id.to_le_bytes());
            buf.extend_from_slice(&f.append_fsm_file_id.to_le_bytes());
        }

        // Append page counts per (branch, table).
        let mut count_entries: Vec<(u64, u32, u64)> = Vec::new();
        self.append_counts.iter_sync(|k, v| {
            count_entries.push(((*k >> 32) as u64, (*k & 0xFFFF_FFFF) as u32, *v));
            true
        });
        buf.extend_from_slice(&(count_entries.len() as u32).to_le_bytes());
        for (bid, hfid, c) in &count_entries {
            buf.extend_from_slice(&bid.to_le_bytes());
            buf.extend_from_slice(&hfid.to_le_bytes());
            buf.extend_from_slice(&c.to_le_bytes());
        }

        // Page overrides (original main page -> branch cow page) per branch.
        let mut branch_ids: Vec<u64> = Vec::new();
        self.page_trackers.iter_sync(|k, _| {
            branch_ids.push(*k);
            true
        });
        let mut ov_entries: Vec<(u64, PageId, PageId)> = Vec::new();
        for bid in &branch_ids {
            self.page_trackers.read_sync(bid, |_, tracker| {
                tracker.for_each_override(|orig, local| ov_entries.push((*bid, orig, local)));
            });
        }
        buf.extend_from_slice(&(ov_entries.len() as u32).to_le_bytes());
        for (bid, orig, local) in &ov_entries {
            buf.extend_from_slice(&bid.to_le_bytes());
            write_page_id(&mut buf, *orig);
            write_page_id(&mut buf, *local);
        }

        let path = self.state_path();
        let tmp = path.with_extension("zybranches.tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(ZyronError::Io)?;
            use std::io::Write;
            f.write_all(&buf).map_err(ZyronError::Io)?;
            f.sync_all().map_err(ZyronError::Io)?;
        }
        std::fs::rename(&tmp, &path).map_err(ZyronError::Io)?;
        Ok(())
    }

    /// Loads branch metadata from disk, repopulating the branch map and the id
    /// allocator. Page-override trackers start empty (overrides are rebuilt by
    /// the storage recovery path). Missing or empty file is a clean no-op.
    pub fn load(&self) -> Result<()> {
        let path = self.state_path();
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(ZyronError::Io(e)),
        };
        if data.len() < 4 {
            return Ok(());
        }
        let mut off = 0usize;
        let count = read_u32(&data, &mut off)? as usize;
        let mut max_id = 0u64;
        for _ in 0..count {
            let id = read_u64(&data, &mut off)?;
            let has_parent = read_u8(&data, &mut off)?;
            let parent = if has_parent != 0 {
                Some(BranchId(read_u64(&data, &mut off)?))
            } else {
                None
            };
            let base_version = read_u64(&data, &mut off)?;
            let created_at = read_i64(&data, &mut off)?;
            let name = read_str(&data, &mut off)?;
            let description = read_str(&data, &mut off)?;
            max_id = max_id.max(id);
            let entry = BranchEntry {
                id: BranchId(id),
                name,
                parent_branch_id: parent,
                base_version_id: VersionId(base_version),
                created_at,
                description,
                is_active: true,
            };
            let _ = self.branches.insert_sync(id, entry);
            let _ = self
                .page_trackers
                .insert_sync(id, BranchPageTracker::new(1024));
        }
        self.next_branch_id.store(max_id + 1, Ordering::Relaxed);

        // Older state files end after the branch section and carry no overlay.
        if off >= data.len() {
            return Ok(());
        }

        // File-id allocator high-water mark.
        let nbf = read_u32(&data, &mut off)?;
        self.next_branch_file_id.store(nbf, Ordering::Relaxed);

        // Branch overlay file ids.
        let files_count = read_u32(&data, &mut off)? as usize;
        for _ in 0..files_count {
            let bid = read_u64(&data, &mut off)?;
            let hfid = read_u32(&data, &mut off)?;
            let cow_file_id = read_u32(&data, &mut off)?;
            let append_file_id = read_u32(&data, &mut off)?;
            let append_fsm_file_id = read_u32(&data, &mut off)?;
            let _ = self.branch_table_files.insert_sync(
                branch_table_key(bid, hfid),
                zyron_common::BranchFiles {
                    cow_file_id,
                    append_file_id,
                    append_fsm_file_id,
                },
            );
        }

        // Append page counts.
        let counts = read_u32(&data, &mut off)? as usize;
        for _ in 0..counts {
            let bid = read_u64(&data, &mut off)?;
            let hfid = read_u32(&data, &mut off)?;
            let c = read_u64(&data, &mut off)?;
            let _ = self
                .append_counts
                .insert_sync(branch_table_key(bid, hfid), c);
        }

        // Page overrides.
        let overrides = read_u32(&data, &mut off)? as usize;
        for _ in 0..overrides {
            let bid = read_u64(&data, &mut off)?;
            let orig = read_page_id(&data, &mut off)?;
            let local = read_page_id(&data, &mut off)?;
            self.page_trackers.read_sync(&bid, |_, tracker| {
                tracker.set_modified(orig.page_num, orig, local)
            });
        }

        Ok(())
    }
}

fn write_page_id(buf: &mut Vec<u8>, p: PageId) {
    buf.extend_from_slice(&p.file_id.to_le_bytes());
    buf.extend_from_slice(&p.page_num.to_le_bytes());
}

fn read_page_id(data: &[u8], off: &mut usize) -> Result<PageId> {
    let file_id = read_u32(data, off)?;
    let page_num = read_u64(data, off)?;
    Ok(PageId::new(file_id, page_num))
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn read_u8(data: &[u8], off: &mut usize) -> Result<u8> {
    let v = *data
        .get(*off)
        .ok_or_else(|| ZyronError::Internal("branch state truncated".into()))?;
    *off += 1;
    Ok(v)
}

fn read_u32(data: &[u8], off: &mut usize) -> Result<u32> {
    let end = *off + 4;
    let slice = data
        .get(*off..end)
        .ok_or_else(|| ZyronError::Internal("branch state truncated".into()))?;
    *off = end;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(data: &[u8], off: &mut usize) -> Result<u64> {
    let end = *off + 8;
    let slice = data
        .get(*off..end)
        .ok_or_else(|| ZyronError::Internal("branch state truncated".into()))?;
    *off = end;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn read_i64(data: &[u8], off: &mut usize) -> Result<i64> {
    Ok(read_u64(data, off)? as i64)
}

fn read_str(data: &[u8], off: &mut usize) -> Result<String> {
    let len = read_u32(data, off)? as usize;
    let end = *off + len;
    let slice = data
        .get(*off..end)
        .ok_or_else(|| ZyronError::Internal("branch state truncated".into()))?;
    *off = end;
    String::from_utf8(slice.to_vec())
        .map_err(|e| ZyronError::Internal(format!("branch name utf8: {e}")))
}

impl zyron_common::BranchCatalog for BranchManager {
    fn resolve_page_for(&self, branch_id: u64, page_id: PageId) -> PageId {
        self.resolve_page(BranchId(branch_id), page_id)
    }

    fn branch_id_by_name(&self, name: &str) -> Option<u64> {
        self.get_branch_by_name(name).ok().map(|e| e.id.0)
    }

    fn branch_files_for(&self, branch_id: u64, heap_file_id: u32) -> zyron_common::BranchFiles {
        let key = branch_table_key(branch_id, heap_file_id);
        if let Some(f) = self.branch_table_files.read_sync(&key, |_, f| *f) {
            return f;
        }
        // Three consecutive ids: cow data, append data, append fsm.
        let base = self.next_branch_file_id.fetch_add(3, Ordering::Relaxed);
        let files = zyron_common::BranchFiles {
            cow_file_id: base,
            append_file_id: base + 1,
            append_fsm_file_id: base + 2,
        };
        match self.branch_table_files.insert_sync(key, files) {
            Ok(()) => files,
            // Lost the race, the winner's ids are authoritative; our reserved
            // ids leak harmlessly.
            Err(_) => self
                .branch_table_files
                .read_sync(&key, |_, f| *f)
                .unwrap_or(files),
        }
    }

    fn lookup_cow_page(&self, branch_id: u64, original_page: PageId) -> Option<PageId> {
        self.page_trackers
            .read_sync(&branch_id, |_, tracker| tracker.resolve(original_page))
            .flatten()
    }

    fn record_cow_page(&self, branch_id: u64, original_page: PageId, local_page: PageId) -> PageId {
        self.page_trackers
            .read_sync(&branch_id, |_, tracker| {
                tracker.get_or_insert_override(original_page, local_page)
            })
            .unwrap_or(local_page)
    }

    fn append_page_count(&self, branch_id: u64, heap_file_id: u32) -> u64 {
        self.append_counts
            .read_sync(&branch_table_key(branch_id, heap_file_id), |_, v| *v)
            .unwrap_or(0)
    }

    fn set_append_page_count(&self, branch_id: u64, heap_file_id: u32, count: u64) {
        let key = branch_table_key(branch_id, heap_file_id);
        if self
            .append_counts
            .update_sync(&key, |_, v| *v = count)
            .is_none()
        {
            let _ = self.append_counts.insert_sync(key, count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    fn page(file_id: u32, page_num: u64) -> PageId {
        PageId::new(file_id, page_num)
    }

    #[test]
    fn test_branch_id_display() {
        assert_eq!(BranchId(42).to_string(), "branch_42");
    }

    #[test]
    fn test_create_branch() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let id = mgr
            .create_branch("dev", None, VersionId(10), "development branch", 1000)
            .expect("create");
        assert_eq!(id, BranchId(1));

        let entry = mgr.get_branch(id).expect("get");
        assert_eq!(entry.name, "dev");
        assert_eq!(entry.base_version_id, VersionId(10));
        assert!(entry.is_active);
    }

    #[test]
    fn test_duplicate_branch_name() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        mgr.create_branch("dev", None, VersionId(1), "", 1000)
            .expect("create");
        assert!(
            mgr.create_branch("dev", None, VersionId(2), "", 2000)
                .is_err()
        );
    }

    #[test]
    fn test_resolve_page_no_override() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let id = mgr
            .create_branch("dev", None, VersionId(1), "", 1000)
            .expect("create");

        let p = page(1, 42);
        assert_eq!(mgr.resolve_page(id, p), p);
    }

    #[test]
    fn test_resolve_page_with_override() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let id = mgr
            .create_branch("dev", None, VersionId(1), "", 1000)
            .expect("create");

        let original = page(1, 42);
        let local = page(50001, 0);

        mgr.record_page_override(id, original, local)
            .expect("record");
        assert_eq!(mgr.resolve_page(id, original), local);
        assert_eq!(mgr.modified_page_count(id), 1);
    }

    #[test]
    fn test_resolve_page_parent_chain() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let parent_id = mgr
            .create_branch("main", None, VersionId(1), "", 1000)
            .expect("create parent");
        let child_id = mgr
            .create_branch("dev", Some(parent_id), VersionId(5), "", 2000)
            .expect("create child");

        let original = page(1, 10);
        let parent_local = page(50001, 0);

        // Override in parent
        mgr.record_page_override(parent_id, original, parent_local)
            .expect("record");

        // Child should resolve through to parent's override
        assert_eq!(mgr.resolve_page(child_id, original), parent_local);

        // Child overrides the same page
        let child_local = page(50002, 0);
        mgr.record_page_override(child_id, original, child_local)
            .expect("record child");

        // Now child resolves to its own override
        assert_eq!(mgr.resolve_page(child_id, original), child_local);
        // Parent still has its override
        assert_eq!(mgr.resolve_page(parent_id, original), parent_local);
    }

    #[test]
    fn test_bitset_fast_path() {
        let tracker = BranchPageTracker::new(1024);

        // Unmodified page: bitset confirms definitely unmodified
        assert!(tracker.is_definitely_unmodified(42));

        // Modified page
        let original = page(1, 42);
        let local = page(2, 0);
        tracker.set_modified(42, original, local);
        assert!(!tracker.is_definitely_unmodified(42));
        assert!(tracker.is_definitely_unmodified(43)); // neighbor unaffected
    }

    #[test]
    fn test_merge_no_conflicts() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let main_id = mgr
            .create_branch("main", None, VersionId(1), "", 1000)
            .expect("main");
        let dev_id = mgr
            .create_branch("dev", Some(main_id), VersionId(1), "", 2000)
            .expect("dev");

        // Dev modifies page 10
        mgr.record_page_override(dev_id, page(1, 10), page(50002, 0))
            .expect("record");

        let result = mgr
            .merge_branch(dev_id, main_id, VersionId(5))
            .expect("merge");
        assert_eq!(result.merged_pages, 1);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn test_merge_with_conflict() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let main_id = mgr
            .create_branch("main", None, VersionId(1), "", 1000)
            .expect("main");
        let dev_id = mgr
            .create_branch("dev", Some(main_id), VersionId(1), "", 2000)
            .expect("dev");

        // Both modify the same page
        mgr.record_page_override(main_id, page(1, 10), page(50001, 0))
            .expect("main override");
        mgr.record_page_override(dev_id, page(1, 10), page(50002, 0))
            .expect("dev override");

        let result = mgr
            .merge_branch(dev_id, main_id, VersionId(5))
            .expect("merge");
        assert_eq!(result.merged_pages, 0);
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(
            result.conflicts[0].conflict_type,
            ConflictType::BothModified
        );
    }

    #[test]
    fn test_delete_branch() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let id = mgr
            .create_branch("dev", None, VersionId(1), "", 1000)
            .expect("create");
        assert_eq!(mgr.list_branches().len(), 1);

        mgr.delete_branch(id).expect("delete");
        assert_eq!(mgr.list_branches().len(), 0);
    }

    #[test]
    fn test_list_branches() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        mgr.create_branch("main", None, VersionId(1), "", 1000)
            .expect("main");
        mgr.create_branch("dev", None, VersionId(1), "", 2000)
            .expect("dev");
        mgr.create_branch("feature", None, VersionId(1), "", 3000)
            .expect("feature");

        assert_eq!(mgr.list_branches().len(), 3);
    }

    #[test]
    fn test_get_branch_by_name() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        mgr.create_branch("dev", None, VersionId(1), "dev branch", 1000)
            .expect("create");

        let entry = mgr.get_branch_by_name("dev").expect("find");
        assert_eq!(entry.name, "dev");
        assert_eq!(entry.description, "dev branch");

        assert!(mgr.get_branch_by_name("nonexistent").is_err());
    }

    #[test]
    fn test_branch_file_id_base() {
        assert_eq!(BranchManager::branch_file_id_base(BranchId(1)), 51000);
        assert_eq!(BranchManager::branch_file_id_base(BranchId(5)), 55000);
    }

    #[test]
    fn test_merge_nonexistent_branch() {
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let main_id = mgr
            .create_branch("main", None, VersionId(1), "", 1000)
            .expect("main");

        assert!(
            mgr.merge_branch(BranchId(999), main_id, VersionId(1))
                .is_err()
        );
    }

    #[test]
    fn test_persist_and_reload_branches() {
        let dir = make_temp_dir();
        {
            let mgr = BranchManager::new(dir.path().to_path_buf());
            let dev = mgr
                .create_branch("dev", None, VersionId(5), "dev branch", 1000)
                .expect("dev");
            mgr.create_branch("feature", Some(dev), VersionId(7), "", 2000)
                .expect("feature");
            // Persistence is explicit (the DDL boundary does this in production).
            mgr.persist().expect("persist");
        }
        // A fresh manager over the same dir recovers both branches and the id
        // allocator so the next branch does not reuse an id.
        let reloaded = BranchManager::new(dir.path().to_path_buf());
        reloaded.load().expect("load");
        let names: Vec<String> = reloaded
            .list_branches()
            .into_iter()
            .map(|b| b.name)
            .collect();
        assert!(names.contains(&"dev".to_string()));
        assert!(names.contains(&"feature".to_string()));
        let feature = reloaded.get_branch_by_name("feature").expect("feature");
        assert_eq!(feature.base_version_id, VersionId(7));
        assert!(feature.parent_branch_id.is_some());
        let next = reloaded
            .create_branch("third", None, VersionId(9), "", 3000)
            .expect("third");
        assert_eq!(next, BranchId(3), "id allocator resumed past reloaded max");
    }

    #[test]
    fn test_branch_cow_write_methods() {
        use zyron_common::BranchCatalog;
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());
        let bid = mgr
            .create_branch("dev", None, VersionId(1), "", 1000)
            .expect("dev")
            .0;

        // Overlay file ids are stable and distinct across tables.
        let f200 = mgr.branch_files_for(bid, 200);
        let f201 = mgr.branch_files_for(bid, 201);
        assert_eq!(mgr.branch_files_for(bid, 200), f200, "stable per table");
        assert_ne!(f200.cow_file_id, f201.cow_file_id);
        assert!(f200.cow_file_id >= 2_000_000);
        assert_ne!(f200.cow_file_id, f200.append_file_id);
        assert_ne!(f200.append_file_id, f200.append_fsm_file_id);

        // No cow copy until one is recorded.
        let main_page = page(200, 7);
        assert_eq!(mgr.lookup_cow_page(bid, main_page), None);
        let cow = page(f200.cow_file_id, 0);
        assert_eq!(mgr.record_cow_page(bid, main_page, cow), cow);
        assert_eq!(mgr.lookup_cow_page(bid, main_page), Some(cow));
        // Re-recording returns the first winner, not the new candidate.
        let other = page(f200.cow_file_id, 99);
        assert_eq!(mgr.record_cow_page(bid, main_page, other), cow);
        assert_eq!(mgr.resolve_page_for(bid, main_page), cow);

        // Append page count round-trips.
        assert_eq!(mgr.append_page_count(bid, 200), 0);
        mgr.set_append_page_count(bid, 200, 5);
        assert_eq!(mgr.append_page_count(bid, 200), 5);
        mgr.set_append_page_count(bid, 200, 6);
        assert_eq!(mgr.append_page_count(bid, 200), 6);
        assert_eq!(mgr.append_page_count(bid, 201), 0, "per table");
    }

    #[test]
    fn test_persist_and_reload_overlay() {
        use zyron_common::BranchCatalog;
        let dir = make_temp_dir();
        let orig = page(200, 5);
        let (bid, files, cowpage);
        {
            let mgr = BranchManager::new(dir.path().to_path_buf());
            bid = mgr
                .create_branch("dev", None, VersionId(1), "", 1000)
                .expect("dev");
            files = mgr.branch_files_for(bid.0, 200);
            cowpage = page(files.cow_file_id, 0);
            mgr.record_cow_page(bid.0, orig, cowpage);
            mgr.set_append_page_count(bid.0, 200, 3);
            mgr.persist().expect("persist");
        }
        // A fresh manager recovers the overlay: file-id map, append counts, and
        // page overrides all survive, and the file-id allocator does not reuse
        // an id already handed out.
        let reloaded = BranchManager::new(dir.path().to_path_buf());
        reloaded.load().expect("load");
        assert_eq!(reloaded.branch_files_lookup(bid, 200), Some(files));
        assert_eq!(reloaded.append_pages(bid, 200), 3);
        assert_eq!(
            reloaded.resolve_page(bid, orig),
            cowpage,
            "override resolves after reload"
        );
        let files2 = reloaded.branch_files_for(bid.0, 201);
        assert_ne!(
            files2.cow_file_id, files.cow_file_id,
            "allocator resumed past the reloaded high-water mark"
        );
    }

    #[test]
    fn test_branch_catalog_trait() {
        use zyron_common::BranchCatalog;
        let dir = make_temp_dir();
        let mgr = BranchManager::new(dir.path().to_path_buf());
        let id = mgr
            .create_branch("dev", None, VersionId(1), "", 1000)
            .expect("dev");
        // Name resolution through the trait the executor uses.
        assert_eq!(mgr.branch_id_by_name("dev"), Some(id.0));
        assert_eq!(mgr.branch_id_by_name("missing"), None);
        // With no override, resolve returns the same page.
        let pid = page(200, 3);
        assert_eq!(mgr.resolve_page_for(id.0, pid), pid);
        // After recording an override, the trait returns the branch-local page.
        let local = page(50000, 9);
        mgr.record_page_override(id, pid, local).expect("override");
        assert_eq!(mgr.resolve_page_for(id.0, pid), local);
    }
}

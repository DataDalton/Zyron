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
        let word_idx = (page_num / 64) as usize;
        if word_idx < self.modified_bitset.len() {
            let bit = 1u64 << (page_num % 64);
            self.modified_bitset[word_idx].fetch_or(bit, Ordering::Relaxed);
        }
        // Always insert into HashMap regardless of bitset capacity
        let _ = self.overrides.insert_sync(original, local);
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
    next_branch_id: AtomicU64,
    data_dir: PathBuf,
}

impl BranchManager {
    /// Creates a new branch manager.
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            branches: scc::HashMap::new(),
            page_trackers: scc::HashMap::new(),
            next_branch_id: AtomicU64::new(1),
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

    /// Deletes a branch and its page overrides.
    pub fn delete_branch(&self, branch_id: BranchId) -> Result<()> {
        let removed = self.branches.update_sync(&branch_id.0, |_, entry| {
            entry.is_active = false;
        });
        if removed.is_none() {
            return Err(ZyronError::BranchNotFound(format!("{}", branch_id)));
        }
        let _ = self.page_trackers.remove_sync(&branch_id.0);
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
}

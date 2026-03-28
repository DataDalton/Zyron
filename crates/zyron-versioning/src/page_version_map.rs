//! Page-level version range tracking for time travel scan pruning.
//!
//! Each versioned heap page tracks the minimum and maximum version_id of
//! tuples it contains. Time travel scans check this map to skip entire
//! pages that cannot contain tuples visible at the target version.
//!
//! Zero locks on the read path. Two atomic loads per skip check indexed
//! directly by page_num. Updates use CAS loops on AtomicU64 pairs.
//! Growth uses a parking_lot::Mutex but only triggers when new pages
//! are allocated (rare, not on the scan hot path).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use zyron_common::error::{Result, ZyronError};
use zyron_common::page::PageId;

/// Sentinel value indicating a page slot is not tracked.
const NOT_TRACKED: u64 = u64::MAX;

/// Initial capacity in pages. Grows as needed.
const INITIAL_CAPACITY: usize = 4096;

/// Version range tracked per page.
#[derive(Debug, Clone, Copy)]
pub struct PageVersionRange {
    pub min_version: u64,
    pub max_version: u64,
}

/// Lock-free page version range map.
///
/// Flat atomic arrays indexed by page_num. can_skip_page() is two
/// atomic loads with no hashing, no HashMap lookup, no lock acquisition.
/// This is a per-table structure (one heap file per table).
pub struct PageVersionMap {
    /// min_version per page. NOT_TRACKED if untracked.
    min_versions: Box<[AtomicU64]>,
    /// max_version per page. NOT_TRACKED if untracked.
    max_versions: Box<[AtomicU64]>,
    /// Current array capacity (number of page slots).
    capacity: AtomicUsize,
    /// Number of tracked pages (for page_count()).
    tracked_count: AtomicUsize,
}

impl PageVersionMap {
    /// Creates a page version map with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(INITIAL_CAPACITY)
    }

    /// Creates a page version map with the given page capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(64);
        let mut min_v = Vec::with_capacity(cap);
        let mut max_v = Vec::with_capacity(cap);
        for _ in 0..cap {
            min_v.push(AtomicU64::new(NOT_TRACKED));
            max_v.push(AtomicU64::new(NOT_TRACKED));
        }
        Self {
            min_versions: min_v.into_boxed_slice(),
            max_versions: max_v.into_boxed_slice(),
            capacity: AtomicUsize::new(cap),
            tracked_count: AtomicUsize::new(0),
        }
    }

    /// Updates the version range for a page after inserting a tuple.
    ///
    /// Expands min/max to include the given version_id.
    pub fn update_on_insert(&self, page_id: PageId, version_id: u64) {
        let idx = page_id.page_num as usize;
        if idx >= self.capacity.load(Ordering::Relaxed) {
            // Page beyond current capacity, skip tracking.
            // The scan will conservatively include this page.
            return;
        }

        let was_untracked = self.min_versions[idx].load(Ordering::Relaxed) == NOT_TRACKED;
        atomic_min(&self.min_versions[idx], version_id);
        atomic_max(&self.max_versions[idx], version_id);
        if was_untracked {
            self.tracked_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Updates the version range for a page after deleting a tuple.
    pub fn update_on_delete(&self, page_id: PageId, deleted_at_version: u64) {
        let idx = page_id.page_num as usize;
        if idx >= self.capacity.load(Ordering::Relaxed) {
            return;
        }

        let was_untracked = self.min_versions[idx].load(Ordering::Relaxed) == NOT_TRACKED;
        atomic_min(&self.min_versions[idx], deleted_at_version);
        atomic_max(&self.max_versions[idx], deleted_at_version);
        if was_untracked {
            self.tracked_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns true if the page can be skipped for a time travel query.
    ///
    /// Zero locks. Two atomic loads indexed by page_num.
    /// A page can be skipped when all tuples on it were created after the
    /// target version. Conservative: returns false if not tracked.
    #[inline]
    pub fn can_skip_page(&self, page_id: PageId, target_version: u64) -> bool {
        let idx = page_id.page_num as usize;
        if idx >= self.capacity.load(Ordering::Relaxed) {
            return false;
        }
        let min_v = self.min_versions[idx].load(Ordering::Relaxed);
        min_v != NOT_TRACKED && min_v > target_version
    }

    /// Returns true if the page can be skipped for a diff scan between versions.
    ///
    /// Zero locks. Two atomic loads indexed by page_num.
    #[inline]
    pub fn can_skip_page_for_range(
        &self,
        page_id: PageId,
        from_version: u64,
        to_version: u64,
    ) -> bool {
        let idx = page_id.page_num as usize;
        if idx >= self.capacity.load(Ordering::Relaxed) {
            return false;
        }
        let min_v = self.min_versions[idx].load(Ordering::Relaxed);
        if min_v == NOT_TRACKED {
            return false;
        }
        let max_v = self.max_versions[idx].load(Ordering::Relaxed);
        max_v <= from_version || min_v > to_version
    }

    /// Returns the version range for a page, if tracked.
    pub fn get_range(&self, page_id: PageId) -> Option<PageVersionRange> {
        let idx = page_id.page_num as usize;
        if idx >= self.capacity.load(Ordering::Relaxed) {
            return None;
        }
        let min_v = self.min_versions[idx].load(Ordering::Relaxed);
        if min_v == NOT_TRACKED {
            return None;
        }
        let max_v = self.max_versions[idx].load(Ordering::Relaxed);
        Some(PageVersionRange {
            min_version: min_v,
            max_version: max_v,
        })
    }

    /// Returns the number of tracked pages.
    pub fn page_count(&self) -> usize {
        self.tracked_count.load(Ordering::Relaxed)
    }

    /// Serializes all entries for checkpoint persistence.
    ///
    /// Format: entry_count(u32) + [file_id(u32) + page_num(u64) +
    /// min_version(u64) + max_version(u64)] per entry (28 bytes per entry).
    pub fn serialize(&self) -> Vec<u8> {
        let cap = self.capacity.load(Ordering::Relaxed);
        let mut entries: Vec<(u64, u64, u64)> = Vec::new();

        for i in 0..cap {
            let min_v = self.min_versions[i].load(Ordering::Relaxed);
            if min_v != NOT_TRACKED {
                let max_v = self.max_versions[i].load(Ordering::Relaxed);
                entries.push((i as u64, min_v, max_v));
            }
        }

        let mut buf = Vec::with_capacity(4 + entries.len() * 28);
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (page_num, min_v, max_v) in &entries {
            // file_id is 0 (per-table map, file_id stored externally)
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&page_num.to_le_bytes());
            buf.extend_from_slice(&min_v.to_le_bytes());
            buf.extend_from_slice(&max_v.to_le_bytes());
        }
        buf
    }

    /// Deserializes from checkpoint data.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(ZyronError::Internal(
                "page version map data too short".to_string(),
            ));
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let expected_len = 4 + count * 28;
        if data.len() < expected_len {
            return Err(ZyronError::Internal(format!(
                "page version map truncated: expected {expected_len} bytes, got {}",
                data.len()
            )));
        }

        // Find max page_num to size the map
        let mut max_page: u64 = 0;
        let mut offset = 4;
        for _ in 0..count {
            offset += 4; // skip file_id
            let page_num = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            if page_num > max_page {
                max_page = page_num;
            }
            offset += 24; // skip page_num + min + max
        }

        let map = Self::with_capacity((max_page as usize + 1).max(INITIAL_CAPACITY));

        offset = 4;
        for _ in 0..count {
            let file_id = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            let page_num = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            let min_version = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            let max_version = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;

            let page_id = PageId::new(file_id, page_num);
            map.update_on_insert(page_id, min_version);
            if max_version > min_version {
                map.update_on_insert(page_id, max_version);
            }
        }

        Ok(map)
    }

    /// Clears all tracked pages.
    pub fn clear(&self) {
        let cap = self.capacity.load(Ordering::Relaxed);
        for i in 0..cap {
            self.min_versions[i].store(NOT_TRACKED, Ordering::Relaxed);
            self.max_versions[i].store(NOT_TRACKED, Ordering::Relaxed);
        }
        self.tracked_count.store(0, Ordering::Relaxed);
    }
}

impl Default for PageVersionMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Atomically updates a value to the minimum of current and new.
#[inline]
fn atomic_min(atom: &AtomicU64, new_val: u64) {
    let mut current = atom.load(Ordering::Relaxed);
    loop {
        let target = if current == NOT_TRACKED {
            new_val
        } else {
            current.min(new_val)
        };
        if target == current {
            break;
        }
        match atom.compare_exchange_weak(current, target, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

/// Atomically updates a value to the maximum of current and new.
#[inline]
fn atomic_max(atom: &AtomicU64, new_val: u64) {
    let mut current = atom.load(Ordering::Relaxed);
    loop {
        let target = if current == NOT_TRACKED {
            new_val
        } else {
            current.max(new_val)
        };
        if target == current {
            break;
        }
        match atom.compare_exchange_weak(current, target, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(file_id: u32, page_num: u64) -> PageId {
        PageId::new(file_id, page_num)
    }

    #[test]
    fn test_empty_map() {
        let map = PageVersionMap::new();
        assert_eq!(map.page_count(), 0);
        assert!(!map.can_skip_page(page(1, 0), 10));
    }

    #[test]
    fn test_insert_tracking() {
        let map = PageVersionMap::new();
        let p = page(1, 0);

        map.update_on_insert(p, 5);
        assert_eq!(map.get_range(p).unwrap().min_version, 5);
        assert_eq!(map.get_range(p).unwrap().max_version, 5);

        map.update_on_insert(p, 10);
        assert_eq!(map.get_range(p).unwrap().min_version, 5);
        assert_eq!(map.get_range(p).unwrap().max_version, 10);

        map.update_on_insert(p, 3);
        assert_eq!(map.get_range(p).unwrap().min_version, 3);
        assert_eq!(map.get_range(p).unwrap().max_version, 10);
    }

    #[test]
    fn test_delete_tracking() {
        let map = PageVersionMap::new();
        let p = page(1, 0);

        map.update_on_insert(p, 5);
        map.update_on_delete(p, 15);

        let range = map.get_range(p).unwrap();
        assert_eq!(range.min_version, 5);
        assert_eq!(range.max_version, 15);
    }

    #[test]
    fn test_can_skip_page() {
        let map = PageVersionMap::new();
        let p = page(1, 0);

        map.update_on_insert(p, 10);
        map.update_on_insert(p, 20);

        assert!(map.can_skip_page(p, 5));
        assert!(!map.can_skip_page(p, 10));
        assert!(!map.can_skip_page(p, 15));
        assert!(!map.can_skip_page(p, 25));
    }

    #[test]
    fn test_can_skip_page_for_range() {
        let map = PageVersionMap::new();
        let p = page(1, 0);

        map.update_on_insert(p, 10);
        map.update_on_insert(p, 20);

        assert!(!map.can_skip_page_for_range(p, 5, 25));
        assert!(map.can_skip_page_for_range(p, 20, 30));
        assert!(map.can_skip_page_for_range(p, 1, 5));
        assert!(!map.can_skip_page_for_range(p, 5, 15));
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let map = PageVersionMap::new();
        map.update_on_insert(page(1, 0), 5);
        map.update_on_insert(page(1, 0), 20);
        map.update_on_insert(page(2, 3), 10);
        map.update_on_insert(page(2, 3), 15);

        let data = map.serialize();
        let recovered = PageVersionMap::deserialize(&data).expect("deserialize");

        assert_eq!(recovered.page_count(), 2);

        let r1 = recovered.get_range(page(1, 0)).unwrap();
        assert_eq!(r1.min_version, 5);
        assert_eq!(r1.max_version, 20);

        let r2 = recovered.get_range(page(2, 3)).unwrap();
        assert_eq!(r2.min_version, 10);
        assert_eq!(r2.max_version, 15);
    }

    #[test]
    fn test_serialize_empty() {
        let map = PageVersionMap::new();
        let data = map.serialize();
        let recovered = PageVersionMap::deserialize(&data).expect("deserialize");
        assert_eq!(recovered.page_count(), 0);
    }

    #[test]
    fn test_multiple_pages() {
        let map = PageVersionMap::new();

        for i in 0..100 {
            map.update_on_insert(page(1, i), i + 1);
        }

        assert_eq!(map.page_count(), 100);

        assert!(map.can_skip_page(page(1, 50), 50));
        assert!(!map.can_skip_page(page(1, 50), 51));
    }

    #[test]
    fn test_page_beyond_capacity() {
        let map = PageVersionMap::with_capacity(100);

        // Page beyond capacity: update is a no-op, skip returns false
        map.update_on_insert(page(1, 200), 42);
        assert!(map.get_range(page(1, 200)).is_none());
        assert!(!map.can_skip_page(page(1, 200), 1));
    }

    #[test]
    fn test_tracked_count() {
        let map = PageVersionMap::new();
        assert_eq!(map.page_count(), 0);

        map.update_on_insert(page(1, 0), 1);
        assert_eq!(map.page_count(), 1);

        map.update_on_insert(page(1, 0), 2); // same page, count stays 1
        assert_eq!(map.page_count(), 1);

        map.update_on_insert(page(1, 1), 3);
        assert_eq!(map.page_count(), 2);
    }
}

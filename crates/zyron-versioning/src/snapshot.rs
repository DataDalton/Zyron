//! Snapshot reader for time travel queries.
//!
//! Reconstructs table state at a specific version or timestamp by scanning
//! heap pages with version-based filtering. Uses PageVersionMap to skip
//! pages that cannot contain visible tuples.

use std::sync::Arc;

use zyron_common::error::Result;
use zyron_common::page::PageId;

use crate::page_version_map::PageVersionMap;
use crate::version::{VersionId, VersionLog};

/// Snapshot of a table at a specific version.
#[derive(Debug, Clone)]
pub struct TableSnapshot {
    /// Version this snapshot represents.
    pub version_id: VersionId,
    /// Commit timestamp of the version (microseconds since epoch).
    pub timestamp: i64,
    /// Number of visible rows at this version.
    pub row_count: u64,
    /// Pages that contain at least one visible tuple at this version.
    pub page_ids: Vec<PageId>,
}

/// Reads table state at a specific version or timestamp.
///
/// Uses the version log to resolve timestamps to version IDs and the
/// page version map to skip pages outside the target version range.
pub struct SnapshotReader {
    table_id: u32,
    heap_file_id: u32,
    version_log: Arc<VersionLog>,
    page_version_map: Arc<PageVersionMap>,
}

impl SnapshotReader {
    /// Creates a new snapshot reader for a table.
    pub fn new(
        table_id: u32,
        heap_file_id: u32,
        version_log: Arc<VersionLog>,
        page_version_map: Arc<PageVersionMap>,
    ) -> Self {
        Self {
            table_id,
            heap_file_id,
            version_log,
            page_version_map,
        }
    }

    /// Returns the table_id this reader targets.
    pub fn table_id(&self) -> u32 {
        self.table_id
    }

    /// Returns the heap_file_id for this table.
    pub fn heap_file_id(&self) -> u32 {
        self.heap_file_id
    }

    /// Returns a reference to the version log.
    pub fn version_log(&self) -> &VersionLog {
        &self.version_log
    }

    /// Returns a reference to the page version map.
    pub fn page_version_map(&self) -> &PageVersionMap {
        &self.page_version_map
    }

    /// Resolves a version_id, validating it exists in the log.
    pub fn resolve_version(&self, version_id: VersionId) -> Result<VersionId> {
        let entry = self.version_log.get_version(version_id)?;
        Ok(entry.version_id)
    }

    /// Resolves a timestamp to the version at or before that time.
    pub fn resolve_timestamp(&self, timestamp: i64) -> Result<VersionId> {
        let entry = self.version_log.get_version_at_timestamp(timestamp)?;
        Ok(entry.version_id)
    }

    /// Checks if a page can be skipped for a time travel query.
    #[inline]
    pub fn can_skip_page(&self, page_id: PageId, target_version: u64) -> bool {
        self.page_version_map.can_skip_page(page_id, target_version)
    }

    /// Returns true if a tuple with the given version bounds is visible
    /// at the target version.
    #[inline]
    pub fn is_tuple_visible_at_version(
        version_id: u64,
        deleted_at_version: u64,
        target: u64,
    ) -> bool {
        version_id <= target && (deleted_at_version == 0 || deleted_at_version > target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::OperationType;

    fn make_temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    #[test]
    fn test_tuple_visibility() {
        // Live tuple at version 5
        assert!(SnapshotReader::is_tuple_visible_at_version(5, 0, 5));
        assert!(SnapshotReader::is_tuple_visible_at_version(5, 0, 10));
        assert!(!SnapshotReader::is_tuple_visible_at_version(5, 0, 4));

        // Deleted tuple: created at 5, deleted at 10
        assert!(SnapshotReader::is_tuple_visible_at_version(5, 10, 5));
        assert!(SnapshotReader::is_tuple_visible_at_version(5, 10, 9));
        assert!(!SnapshotReader::is_tuple_visible_at_version(5, 10, 10));
        assert!(!SnapshotReader::is_tuple_visible_at_version(5, 10, 15));
        assert!(!SnapshotReader::is_tuple_visible_at_version(5, 10, 4));
    }

    #[test]
    fn test_snapshot_reader_resolve_version() {
        let dir = make_temp_dir();
        let log = Arc::new(VersionLog::open(dir.path(), 1).expect("open"));
        log.append(1, 1000, OperationType::Insert, 10, None)
            .expect("append");
        log.append(2, 2000, OperationType::Update, -1, None)
            .expect("append");

        let map = Arc::new(PageVersionMap::new());
        let reader = SnapshotReader::new(1, 200, log, map);

        assert_eq!(
            reader.resolve_version(VersionId(1)).expect("v1"),
            VersionId(1)
        );
        assert_eq!(
            reader.resolve_version(VersionId(2)).expect("v2"),
            VersionId(2)
        );
        assert!(reader.resolve_version(VersionId(3)).is_err());
    }

    #[test]
    fn test_snapshot_reader_resolve_timestamp() {
        let dir = make_temp_dir();
        let log = Arc::new(VersionLog::open(dir.path(), 2).expect("open"));
        log.append(1, 1000, OperationType::Insert, 10, None)
            .expect("append");
        log.append(2, 3000, OperationType::Update, 0, None)
            .expect("append");

        let map = Arc::new(PageVersionMap::new());
        let reader = SnapshotReader::new(2, 202, log, map);

        assert_eq!(
            reader.resolve_timestamp(2000).expect("ts 2000"),
            VersionId(1)
        );
        assert_eq!(
            reader.resolve_timestamp(3000).expect("ts 3000"),
            VersionId(2)
        );
    }

    #[test]
    fn test_snapshot_reader_page_skip() {
        let dir = make_temp_dir();
        let log = Arc::new(VersionLog::open(dir.path(), 3).expect("open"));
        let map = Arc::new(PageVersionMap::new());

        let p1 = PageId::new(1, 0);
        map.update_on_insert(p1, 10);

        let reader = SnapshotReader::new(3, 200, log, map);

        // Page has min_version=10, target=5: can skip
        assert!(reader.can_skip_page(p1, 5));
        // Target=10: cannot skip
        assert!(!reader.can_skip_page(p1, 10));
    }
}

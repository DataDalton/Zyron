//! Table diff and restore operations.
//!
//! Computes the difference between two versions of a table by scanning
//! only pages with modifications in the version range (using PageVersionMap).
//! Restore creates a new version by re-inserting tuples from a past version.

use zyron_storage::TupleId;

use crate::version::VersionId;

// ---------------------------------------------------------------------------
// Diff types
// ---------------------------------------------------------------------------

/// Type of change for a row in a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffType {
    /// Row was added in the target version.
    Added,
    /// Row was removed between the two versions.
    Removed,
    /// Row was modified between the two versions.
    Modified,
}

/// Difference for a single column within a modified row.
#[derive(Debug, Clone)]
pub struct ColumnDiff {
    /// Column index within the row.
    pub column_index: usize,
    /// Old value (None if column was added).
    pub old_value: Option<Vec<u8>>,
    /// New value (None if column was removed).
    pub new_value: Option<Vec<u8>>,
}

/// Difference for a single row.
#[derive(Debug, Clone)]
pub struct RowDiff {
    /// Location of the row in the heap.
    pub tuple_id: TupleId,
    /// Type of change.
    pub diff_type: DiffType,
    /// Column-level differences (only for Modified rows).
    pub column_diffs: Vec<ColumnDiff>,
}

/// Complete diff between two versions of a table.
#[derive(Debug, Clone)]
pub struct TableDiff {
    /// Source version (older).
    pub from_version: VersionId,
    /// Target version (newer).
    pub to_version: VersionId,
    /// Rows added between from and to.
    pub added_rows: Vec<Vec<u8>>,
    /// Rows removed between from and to.
    pub deleted_rows: Vec<Vec<u8>>,
    /// Rows modified between from and to, with column-level diffs.
    pub modified_rows: Vec<RowDiff>,
}

impl TableDiff {
    /// Creates an empty diff between two versions.
    pub fn empty(from: VersionId, to: VersionId) -> Self {
        Self {
            from_version: from,
            to_version: to,
            added_rows: Vec::new(),
            deleted_rows: Vec::new(),
            modified_rows: Vec::new(),
        }
    }

    /// Returns true if the diff contains no changes.
    pub fn is_empty(&self) -> bool {
        self.added_rows.is_empty() && self.deleted_rows.is_empty() && self.modified_rows.is_empty()
    }

    /// Returns summary statistics for this diff.
    pub fn stats(&self) -> DiffStats {
        DiffStats {
            rows_added: self.added_rows.len() as u64,
            rows_removed: self.deleted_rows.len() as u64,
            rows_modified: self.modified_rows.len() as u64,
        }
    }
}

/// Summary statistics for a table diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffStats {
    pub rows_added: u64,
    pub rows_removed: u64,
    pub rows_modified: u64,
}

impl DiffStats {
    /// Total number of changes.
    pub fn total_changes(&self) -> u64 {
        self.rows_added + self.rows_removed + self.rows_modified
    }

    /// Returns true if there are no changes.
    pub fn is_empty(&self) -> bool {
        self.total_changes() == 0
    }
}

impl std::fmt::Display for DiffStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "+{} added, -{} removed, ~{} modified",
            self.rows_added, self.rows_removed, self.rows_modified
        )
    }
}

/// Output format for diff rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffFormat {
    /// Counts only (rows added/removed/modified).
    Summary,
    /// Full row data for each change.
    RowLevel,
    /// Only changed columns for modified rows.
    ColumnLevel,
}

// ---------------------------------------------------------------------------
// Diff computation helpers
// ---------------------------------------------------------------------------

/// Classifies a tuple based on its version bounds relative to a diff range.
///
/// Given a diff range (from_version, to_version], a tuple can be:
/// - Added: created in (from, to] and still live at to
/// - Removed: existed at from but deleted in (from, to]
/// - Unchanged: existed at from and still live at to
pub fn classify_tuple_for_diff(
    version_id: u64,
    deleted_at_version: u64,
    from_version: u64,
    to_version: u64,
) -> Option<DiffType> {
    let visible_at_from = version_id <= from_version
        && (deleted_at_version == 0 || deleted_at_version > from_version);
    let visible_at_to =
        version_id <= to_version && (deleted_at_version == 0 || deleted_at_version > to_version);

    match (visible_at_from, visible_at_to) {
        (false, true) => Some(DiffType::Added),
        (true, false) => Some(DiffType::Removed),
        _ => None, // unchanged or invisible at both versions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::page::PageId;

    #[test]
    fn test_diff_stats_display() {
        let stats = DiffStats {
            rows_added: 10,
            rows_removed: 3,
            rows_modified: 5,
        };
        assert_eq!(stats.to_string(), "+10 added, -3 removed, ~5 modified");
        assert_eq!(stats.total_changes(), 18);
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_diff_stats_empty() {
        let stats = DiffStats {
            rows_added: 0,
            rows_removed: 0,
            rows_modified: 0,
        };
        assert!(stats.is_empty());
        assert_eq!(stats.total_changes(), 0);
    }

    #[test]
    fn test_table_diff_empty() {
        let diff = TableDiff::empty(VersionId(1), VersionId(5));
        assert!(diff.is_empty());
        let stats = diff.stats();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_table_diff_with_changes() {
        let mut diff = TableDiff::empty(VersionId(1), VersionId(5));
        diff.added_rows.push(vec![1, 2, 3]);
        diff.added_rows.push(vec![4, 5, 6]);
        diff.deleted_rows.push(vec![7, 8, 9]);

        assert!(!diff.is_empty());
        let stats = diff.stats();
        assert_eq!(stats.rows_added, 2);
        assert_eq!(stats.rows_removed, 1);
        assert_eq!(stats.rows_modified, 0);
    }

    #[test]
    fn test_classify_tuple_added() {
        // Tuple created at version 5, still live, diff range (3, 10]
        let result = classify_tuple_for_diff(5, 0, 3, 10);
        assert_eq!(result, Some(DiffType::Added));
    }

    #[test]
    fn test_classify_tuple_removed() {
        // Tuple created at version 2, deleted at version 7, diff range (3, 10]
        let result = classify_tuple_for_diff(2, 7, 3, 10);
        assert_eq!(result, Some(DiffType::Removed));
    }

    #[test]
    fn test_classify_tuple_unchanged() {
        // Tuple created at version 2, still live, diff range (3, 10]
        let result = classify_tuple_for_diff(2, 0, 3, 10);
        assert_eq!(result, None); // visible at both, unchanged
    }

    #[test]
    fn test_classify_tuple_invisible_at_both() {
        // Tuple created at version 15, diff range (3, 10]
        let result = classify_tuple_for_diff(15, 0, 3, 10);
        assert_eq!(result, None);
    }

    #[test]
    fn test_classify_tuple_deleted_before_range() {
        // Tuple created at version 1, deleted at version 2, diff range (3, 10]
        let result = classify_tuple_for_diff(1, 2, 3, 10);
        assert_eq!(result, None); // invisible at both
    }

    #[test]
    fn test_classify_tuple_created_and_deleted_in_range() {
        // Tuple created at version 5, deleted at version 8, diff range (3, 10]
        // Visible at from=3? version_id=5 > 3, so not visible at from.
        // Visible at to=10? version_id=5 <= 10 and deleted_at=8 <= 10, so not visible at to.
        let result = classify_tuple_for_diff(5, 8, 3, 10);
        assert_eq!(result, None); // created and deleted within the range
    }

    #[test]
    fn test_column_diff() {
        let cd = ColumnDiff {
            column_index: 2,
            old_value: Some(vec![1, 2]),
            new_value: Some(vec![3, 4]),
        };
        assert_eq!(cd.column_index, 2);
    }

    #[test]
    fn test_row_diff() {
        let rd = RowDiff {
            tuple_id: TupleId::new(PageId::new(1, 0), 5),
            diff_type: DiffType::Modified,
            column_diffs: vec![ColumnDiff {
                column_index: 0,
                old_value: Some(vec![1]),
                new_value: Some(vec![2]),
            }],
        };
        assert_eq!(rd.diff_type, DiffType::Modified);
        assert_eq!(rd.column_diffs.len(), 1);
    }
}

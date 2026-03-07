//! Sorted segment support for primary key columns.
//!
//! During compaction, rows are sorted by primary key before encoding.
//! SortedSegmentIndex stores min/max PK per .zyr file for segment pruning.
//! Binary search within a sorted column enables O(log N) point lookups
//! instead of full segment scans.

use crate::columnar::constants::STAT_VALUE_SIZE;
use crate::columnar::segment::{compare_le_bytes, compare_stat_slots};
use std::path::PathBuf;
use zyron_common::Result;

/// Entry in the sorted segment index, representing one .zyr file.
#[derive(Debug, Clone)]
pub struct SortedSegmentEntry {
    /// Path to the .zyr file.
    pub file_path: PathBuf,
    /// Minimum PK value in this file.
    pub min_pk: [u8; STAT_VALUE_SIZE],
    /// Maximum PK value in this file.
    pub max_pk: [u8; STAT_VALUE_SIZE],
    /// Number of rows in this file (for cost estimation).
    pub row_count: u64,
}

/// Index over sorted .zyr files for segment-level pruning on primary key.
/// Entries are kept sorted by min_pk for binary search.
#[derive(Debug)]
pub struct SortedSegmentIndex {
    entries: Vec<SortedSegmentEntry>,
}

impl SortedSegmentIndex {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a .zyr file to the index. Maintains sorted order by min_pk.
    pub fn add(&mut self, entry: SortedSegmentEntry) {
        let insertPos = self
            .entries
            .binary_search_by(|e| compare_stat_slots(&e.min_pk, &entry.min_pk))
            .unwrap_or_else(|pos| pos);
        self.entries.insert(insertPos, entry);
    }

    /// Removes a .zyr file from the index by path.
    pub fn remove(&mut self, path: &std::path::Path) {
        self.entries.retain(|e| e.file_path != path);
    }

    /// Returns files that might contain the given PK value.
    /// A file is a candidate if min_pk <= value <= max_pk.
    pub fn find_point(&self, pk_value: &[u8; STAT_VALUE_SIZE]) -> Vec<&SortedSegmentEntry> {
        self.entries
            .iter()
            .filter(|e| {
                compare_stat_slots(pk_value, &e.min_pk) != std::cmp::Ordering::Less
                    && compare_stat_slots(pk_value, &e.max_pk) != std::cmp::Ordering::Greater
            })
            .collect()
    }

    /// Returns files that overlap with the PK range [lo, hi].
    pub fn find_range(
        &self,
        lo: &[u8; STAT_VALUE_SIZE],
        hi: &[u8; STAT_VALUE_SIZE],
    ) -> Vec<&SortedSegmentEntry> {
        self.entries
            .iter()
            .filter(|e| {
                compare_stat_slots(&e.max_pk, lo) != std::cmp::Ordering::Less
                    && compare_stat_slots(&e.min_pk, hi) != std::cmp::Ordering::Greater
            })
            .collect()
    }

    /// Returns the total number of indexed files.
    pub fn file_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the total row count across all indexed files.
    pub fn total_rows(&self) -> u64 {
        self.entries.iter().map(|e| e.row_count).sum()
    }
}

/// Binary search within a decoded sorted column for a specific value.
/// Returns the row index if the value is found.
///
/// `decoded_data` contains row_count values of value_size bytes each,
/// stored contiguously in sorted ascending order.
pub fn binary_search_sorted_column(
    decoded_data: &[u8],
    row_count: usize,
    value_size: usize,
    target: &[u8],
) -> Option<usize> {
    if row_count == 0 || target.len() < value_size || decoded_data.len() < row_count * value_size {
        return None;
    }

    let target_slice = &target[..value_size];
    let mut lo = 0usize;
    let mut hi = row_count;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let midOffset = mid * value_size;
        let midValue = &decoded_data[midOffset..midOffset + value_size];

        match compare_le_bytes(midValue, target_slice) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Greater => hi = mid,
        }
    }

    None
}

/// Iterator that merge-scans PK columns across multiple sorted .zyr files.
/// Produces (file_idx, row_idx) pairs in PK order for cross-referencing
/// non-PK columns in the same files.
///
/// Pre-computes all PK values as u64 sort keys at construction time,
/// then uses a flat cursor array with linear min-scan in the hot loop.
/// Linear scan beats BinaryHeap for typical compaction file counts (2-16)
/// due to cache locality and zero allocation per iteration.
pub struct MergeScanIterator {
    /// Pre-computed u64 sort keys per file. sort_keys[file_idx][row_idx].
    sort_keys: Vec<Vec<u64>>,
    /// Row count per file.
    row_counts: Vec<usize>,
    /// Current cursor position per file (next row to emit).
    cursors: Vec<usize>,
    /// Number of active (non-exhausted) files.
    active_count: usize,
}

impl MergeScanIterator {
    /// Creates a merge scan from decoded PK columns of multiple files.
    /// Each entry in `pk_columns` is the decoded PK column for one file.
    pub fn new(
        pk_columns: Vec<Vec<u8>>,
        value_size: usize,
        row_counts: Vec<usize>,
    ) -> Result<Self> {
        let copyLen = value_size.min(8);
        let mut activeCount = 0usize;

        // Pre-compute all sort keys upfront to avoid byte reading in hot loop
        let sortKeys: Vec<Vec<u64>> = pk_columns
            .iter()
            .enumerate()
            .map(|(fileIdx, col)| {
                let rc = row_counts[fileIdx];
                if rc == 0 {
                    return Vec::new();
                }
                activeCount += 1;
                let mut keys = Vec::with_capacity(rc);
                for row in 0..rc {
                    let offset = row * value_size;
                    let end = offset + copyLen;
                    if end <= col.len() {
                        keys.push(read_le_u64(&col[offset..end]));
                    } else {
                        keys.push(u64::MAX);
                    }
                }
                keys
            })
            .collect();

        let cursors = vec![0usize; pk_columns.len()];

        Ok(Self {
            sort_keys: sortKeys,
            row_counts,
            cursors,
            active_count: activeCount,
        })
    }

    /// Advances to the next row in merge order.
    /// Returns (file_idx, row_idx) for the row.
    #[inline]
    pub fn next(&mut self) -> Option<(usize, usize)> {
        if self.active_count == 0 {
            return None;
        }

        // Linear scan for minimum sort key across active cursors.
        let mut bestFile = usize::MAX;
        let mut bestKey = u64::MAX;

        for fileIdx in 0..self.cursors.len() {
            let cursor = self.cursors[fileIdx];
            if cursor < self.row_counts[fileIdx] {
                let key = self.sort_keys[fileIdx][cursor];
                if key < bestKey || (key == bestKey && bestFile == usize::MAX) {
                    bestKey = key;
                    bestFile = fileIdx;
                }
            }
        }

        if bestFile == usize::MAX {
            return None;
        }

        let rowIdx = self.cursors[bestFile];
        self.cursors[bestFile] = rowIdx + 1;

        // Check if this file is now exhausted
        if rowIdx + 1 >= self.row_counts[bestFile] {
            self.active_count -= 1;
        }

        Some((bestFile, rowIdx))
    }
}

/// Reads up to 8 bytes from a slice as a u64 (little-endian).
/// For LE-encoded integer PKs, the resulting u64 preserves numeric ordering.
#[inline(always)]
fn read_le_u64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stat(val: u8) -> [u8; STAT_VALUE_SIZE] {
        let mut stat = [0u8; STAT_VALUE_SIZE];
        stat[0] = val;
        stat
    }

    #[test]
    fn test_sorted_index_add_maintains_order() {
        let mut index = SortedSegmentIndex::new();

        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("c.zyr"),
            min_pk: make_stat(30),
            max_pk: make_stat(40),
            row_count: 100,
        });
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("a.zyr"),
            min_pk: make_stat(10),
            max_pk: make_stat(20),
            row_count: 50,
        });
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("b.zyr"),
            min_pk: make_stat(20),
            max_pk: make_stat(30),
            row_count: 75,
        });

        assert_eq!(index.entries[0].file_path, PathBuf::from("a.zyr"));
        assert_eq!(index.entries[1].file_path, PathBuf::from("b.zyr"));
        assert_eq!(index.entries[2].file_path, PathBuf::from("c.zyr"));
    }

    #[test]
    fn test_find_point() {
        let mut index = SortedSegmentIndex::new();
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("a.zyr"),
            min_pk: make_stat(10),
            max_pk: make_stat(20),
            row_count: 100,
        });
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("b.zyr"),
            min_pk: make_stat(15),
            max_pk: make_stat(30),
            row_count: 100,
        });

        // Value 17 is in both files
        let results = index.find_point(&make_stat(17));
        assert_eq!(results.len(), 2);

        // Value 25 is only in b.zyr
        let results = index.find_point(&make_stat(25));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, PathBuf::from("b.zyr"));

        // Value 5 is in neither
        let results = index.find_point(&make_stat(5));
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_find_range() {
        let mut index = SortedSegmentIndex::new();
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("a.zyr"),
            min_pk: make_stat(10),
            max_pk: make_stat(20),
            row_count: 100,
        });
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("b.zyr"),
            min_pk: make_stat(30),
            max_pk: make_stat(40),
            row_count: 100,
        });

        // Range [15, 35] overlaps both
        let results = index.find_range(&make_stat(15), &make_stat(35));
        assert_eq!(results.len(), 2);

        // Range [25, 28] overlaps neither
        let results = index.find_range(&make_stat(25), &make_stat(28));
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_binary_search_sorted_column() {
        let mut data = Vec::new();
        for v in [10u32, 20, 30, 40, 50] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let target = 30u32.to_le_bytes();
        assert_eq!(binary_search_sorted_column(&data, 5, 4, &target), Some(2));

        let target = 10u32.to_le_bytes();
        assert_eq!(binary_search_sorted_column(&data, 5, 4, &target), Some(0));

        let target = 50u32.to_le_bytes();
        assert_eq!(binary_search_sorted_column(&data, 5, 4, &target), Some(4));

        let target = 25u32.to_le_bytes();
        assert_eq!(binary_search_sorted_column(&data, 5, 4, &target), None);
    }

    #[test]
    fn test_merge_scan_two_files() {
        // File 0: PK values [10, 30, 50]
        let mut col0 = Vec::new();
        for v in [10u32, 30, 50] {
            col0.extend_from_slice(&v.to_le_bytes());
        }

        // File 1: PK values [20, 40, 60]
        let mut col1 = Vec::new();
        for v in [20u32, 40, 60] {
            col1.extend_from_slice(&v.to_le_bytes());
        }

        let mut iter = MergeScanIterator::new(vec![col0, col1], 4, vec![3, 3]).unwrap();

        let expected = vec![
            (0, 0), // 10
            (1, 0), // 20
            (0, 1), // 30
            (1, 1), // 40
            (0, 2), // 50
            (1, 2), // 60
        ];

        for (expectedFile, expectedRow) in expected {
            let (file, row) = iter.next().unwrap();
            assert_eq!(file, expectedFile);
            assert_eq!(row, expectedRow);
        }

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_merge_scan_empty() {
        let mut iter = MergeScanIterator::new(Vec::new(), 4, Vec::new()).unwrap();
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_remove_file() {
        let mut index = SortedSegmentIndex::new();
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("a.zyr"),
            min_pk: make_stat(10),
            max_pk: make_stat(20),
            row_count: 100,
        });
        index.add(SortedSegmentEntry {
            file_path: PathBuf::from("b.zyr"),
            min_pk: make_stat(30),
            max_pk: make_stat(40),
            row_count: 100,
        });

        assert_eq!(index.file_count(), 2);
        index.remove(std::path::Path::new("a.zyr"));
        assert_eq!(index.file_count(), 1);
        assert_eq!(index.entries[0].file_path, PathBuf::from("b.zyr"));
    }
}

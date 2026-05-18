//! Background compaction thread that converts heap rows to .zyr columnar files.
//!
//! Pipeline: materialize rows -> sort by PK -> encode columns (parallel via
//! std::thread::scope) -> build bloom filters -> compute zone maps -> write .zyr file.
//! Runs on a dedicated std::thread with parking_lot::Condvar for wake/sleep.

use std::path::{Path, PathBuf};
use zyron_common::Result;
use zyron_common::types::TypeId;

use crate::columnar::file::{SortOrder, ZyrFileHeader, ZyrFileWriter};
use crate::columnar::segment::ColumnSegment;

/// Configuration for the background compaction thread.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Directory for .zyr output files.
    pub columnar_dir: PathBuf,
    /// Minimum number of committed heap rows before compaction triggers.
    pub min_rows: u64,
    /// Maximum row count per .zyr file.
    pub max_rows_per_file: u64,
    /// Enable fsync after writing .zyr files.
    pub fsync_enabled: bool,
    /// Maximum threads for parallel column encoding.
    /// Clamped to min(column_count, available_cores / 2).
    pub max_encoding_threads: usize,
    /// OLTP p99 latency threshold in microseconds. Compaction pauses
    /// if foreground write latency exceeds this value.
    pub oltp_p99_threshold_us: u64,
    /// Interval between compaction eligibility checks, in milliseconds.
    pub check_interval_ms: u64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            columnar_dir: PathBuf::from("./data/columnar"),
            min_rows: 100_000,
            max_rows_per_file: 1_000_000,
            fsync_enabled: true,
            max_encoding_threads: 4,
            oltp_p99_threshold_us: 1000,
            check_interval_ms: 5000,
        }
    }
}

/// Determines the thread count for parallel column encoding.
/// Uses min(num_columns, available_cores / 2, max_encoding_threads).
pub fn encoding_thread_count(num_columns: usize, max_threads: usize) -> usize {
    let availableCores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(2);
    let halfCores = (availableCores / 2).max(1);
    num_columns.min(halfCores).min(max_threads).max(1)
}

/// Describes a single column for compaction input.
#[derive(Debug, Clone)]
pub struct ColumnDescriptor {
    /// Column ordinal in the table schema.
    pub column_id: u32,
    /// Data type of this column.
    pub type_id: TypeId,
    /// Fixed byte width of each value.
    pub value_size: usize,
    /// Whether this column is the primary key.
    pub is_primary_key: bool,
}

/// Input data for a compaction cycle. Callers materialize heap rows
/// into this structure before calling `run_compaction_cycle`.
pub struct CompactionInput {
    /// Column schema descriptors.
    pub columns: Vec<ColumnDescriptor>,
    /// Per-column row data. `column_data[col_idx]` holds all row values
    /// for that column in row order. Each inner Vec<u8> is one value's bytes.
    /// None represents a null value.
    pub column_data: Vec<Vec<Option<Vec<u8>>>>,
    /// Table identifier.
    pub table_id: u64,
    /// MVCC xmin range (lowest xmin among compacted rows).
    pub xmin_lo: u64,
    /// MVCC xmin range (highest xmin among compacted rows).
    pub xmin_hi: u64,
}

/// Result of a compaction cycle.
#[derive(Debug)]
pub struct CompactionResult {
    /// Path to the written .zyr file.
    pub file_path: PathBuf,
    /// Size of the output file in bytes.
    pub file_size: u64,
    /// Number of rows compacted.
    pub row_count: u64,
    /// Number of columns encoded.
    pub column_count: u32,
}

/// Applies a permutation to a vector in-place using the cycle-leader algorithm.
/// After this call, `data[i]` holds the value that was originally at `perm[i]`.
/// Modifies `perm` in place (marks visited elements) but restores it on first use
/// per column. For subsequent columns on the same permutation, pass a clone.
///
/// Uses only O(1) extra memory (one `Option<Vec<u8>>` temporary) instead of
/// cloning every element.
fn apply_permutation_in_place<T>(data: &mut [T], perm: &mut [usize]) {
    let n = data.len();
    for i in 0..n {
        if perm[i] == i {
            continue;
        }

        let mut current = i;
        loop {
            let target = perm[current];
            perm[current] = current; // Mark as placed
            if target == i {
                break;
            }
            data.swap(current, target);
            current = target;
        }
    }
    // Restore perm for the next column by rebuilding from data positions.
    // Since all cycles are completed, perm[i] == i for all i after processing.
}

/// Runs a single compaction cycle: encodes column data in parallel,
/// writes a .zyr file to the output directory.
///
/// Steps:
/// 1. Optionally sort rows by primary key column.
/// 2. Encode each column in parallel using std::thread::scope.
/// 3. Write the .zyr file with ZyrFileWriter.
///
/// Returns the path and size of the output file.
pub fn run_compaction_cycle(
    config: &CompactionConfig,
    input: CompactionInput,
) -> Result<CompactionResult> {
    let rowCount = if input.column_data.is_empty() {
        0
    } else {
        input.column_data[0].len()
    };

    if rowCount == 0 || input.columns.is_empty() {
        return Err(zyron_common::ZyronError::CompactionFailed(
            "no data to compact".to_string(),
        ));
    }

    // Find primary key column for sorting (if any)
    let pkIndex = input.columns.iter().position(|c| c.is_primary_key);

    // Build sort permutation by PK if present
    let sortedIndices: Vec<usize> = if let Some(pkIdx) = pkIndex {
        let pkData = &input.column_data[pkIdx];
        let mut indices: Vec<usize> = (0..rowCount).collect();
        indices.sort_by(|&a, &b| {
            let va = pkData[a].as_deref().unwrap_or(&[]);
            let vb = pkData[b].as_deref().unwrap_or(&[]);
            crate::columnar::segment::compare_le_bytes(va, vb)
        });
        indices
    } else {
        (0..rowCount).collect()
    };

    // The fold path assigns sys_rowid monotonically and sorts by it, so the
    // permutation is the identity. Detecting that lets every column skip the
    // O(rows) permutation clone and shuffle, and turns the materialized data
    // straight into encoder views with no reordering.
    let identityPerm = sortedIndices.iter().enumerate().all(|(i, &x)| i == x);
    let mut reorderedColumns = input.column_data;
    let columns = input.columns;
    let needs_perm = pkIndex.is_some() && !identityPerm;

    if needs_perm {
        // Permute owned data once per column before borrowing views.
        if encoding_thread_count(columns.len(), config.max_encoding_threads) <= 1
            || columns.len() <= 1
        {
            for colData in reorderedColumns.iter_mut() {
                let mut perm = sortedIndices.clone();
                apply_permutation_in_place(colData, &mut perm);
            }
        } else {
            std::thread::scope(|s| {
                let perm_template = &sortedIndices;
                let handles: Vec<_> = reorderedColumns
                    .iter_mut()
                    .map(|colData| {
                        s.spawn(move || {
                            let mut perm = perm_template.clone();
                            apply_permutation_in_place(colData, &mut perm);
                        })
                    })
                    .collect();
                for h in handles {
                    let _ = h.join();
                }
            });
        }
    }

    let rowCount = if reorderedColumns.is_empty() {
        0
    } else {
        reorderedColumns[0].len()
    };

    encode_and_write(
        config,
        &columns,
        rowCount,
        |i| reorderedColumns[i].iter().map(|v| v.as_deref()).collect(),
        pkIndex,
        input.table_id,
        input.xmin_lo,
        input.xmin_hi,
    )
}

/// Encodes already-ordered columns in parallel and writes the .zyr file.
/// The caller guarantees rows are in final (primary-key) order; this does
/// no sorting. `column_view(i)` yields column `i`'s borrowed value view and
/// is invoked inside that column's own encode worker, so view
/// materialization is parallelized across columns alongside the encode (the
/// serial collect of every column's view before encoding was a regression).
/// `pk_index`, when set, marks the file header sorted Asc on that column.
/// Used by both the legacy owned-data path and the fold's arena-backed
/// zero-per-cell path.
pub fn encode_and_write<'a, V>(
    config: &CompactionConfig,
    columns: &[ColumnDescriptor],
    row_count: usize,
    column_view: V,
    pk_index: Option<usize>,
    table_id: u64,
    xmin_lo: u64,
    xmin_hi: u64,
) -> Result<CompactionResult>
where
    V: Fn(usize) -> Vec<Option<&'a [u8]>> + Sync,
{
    if columns.is_empty() || row_count == 0 {
        return Err(zyron_common::ZyronError::CompactionFailed(
            "no data to compact".to_string(),
        ));
    }
    let rowCount = row_count;

    let threadCount = encoding_thread_count(columns.len(), config.max_encoding_threads);

    let segments: Vec<Result<ColumnSegment>> = if threadCount <= 1 || columns.len() <= 1 {
        columns
            .iter()
            .enumerate()
            .map(|(i, col)| {
                let values = column_view(i);
                ColumnSegment::build(col.column_id, col.type_id, col.value_size, &values)
            })
            .collect()
    } else {
        std::thread::scope(|s| {
            let column_view = &column_view;
            let handles: Vec<_> = columns
                .iter()
                .enumerate()
                .map(|(i, col)| {
                    s.spawn(move || {
                        let values = column_view(i);
                        ColumnSegment::build(col.column_id, col.type_id, col.value_size, &values)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(zyron_common::ZyronError::CompactionFailed(
                            "encoding thread panicked".to_string(),
                        ))
                    })
                })
                .collect()
        })
    };

    let builtSegments: Vec<ColumnSegment> = segments.into_iter().collect::<Result<Vec<_>>>()?;

    let sortOrder = if pk_index.is_some() {
        SortOrder::Asc
    } else {
        SortOrder::None
    };
    let pkColumnId = pk_index.map(|i| columns[i].column_id).unwrap_or(0);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let fileName = format!("table_{}_{}_{}.zyr", table_id, rowCount, timestamp);
    let outputPath = config.columnar_dir.join(&fileName);

    if let Some(parent) = outputPath.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            zyron_common::ZyronError::CompactionFailed(format!(
                "failed to create columnar directory: {}",
                e
            ))
        })?;
    }

    let header = ZyrFileHeader {
        format_version: crate::columnar::constants::ZYR_FORMAT_VERSION,
        column_count: columns.len() as u32,
        row_count: rowCount as u64,
        table_id,
        xmin_range_lo: xmin_lo,
        xmin_range_hi: xmin_hi,
        xmax_range_lo: 0,
        xmax_range_hi: 0,
        primary_key_column_id: pkColumnId,
        sort_order: sortOrder,
    };

    let fileSize = write_zyr_file(&outputPath, header, &builtSegments, config.fsync_enabled)?;

    Ok(CompactionResult {
        file_path: outputPath,
        file_size: fileSize,
        row_count: rowCount as u64,
        column_count: columns.len() as u32,
    })
}

/// Writes a .zyr file from encoded column segments.
fn write_zyr_file(
    path: &Path,
    header: ZyrFileHeader,
    segments: &[ColumnSegment],
    fsync: bool,
) -> Result<u64> {
    let mut writer = ZyrFileWriter::create(path, header)?;

    for segment in segments {
        let headerBytes = segment.header.to_bytes();

        // Serialize bloom filter if present
        let bloomBytes = segment.bloom_filter.as_ref().map(|bf| bf.to_bytes());
        let bloomSlice = bloomBytes.as_deref();

        // Serialize zone maps
        let mut zoneMapBytes = Vec::with_capacity(
            segment.zone_maps.len() * crate::columnar::constants::ZONE_MAP_ENTRY_SIZE,
        );
        for zm in &segment.zone_maps {
            zoneMapBytes.extend_from_slice(&zm.to_bytes());
        }

        writer.write_segment(
            segment.header.column_id,
            &headerBytes,
            bloomSlice,
            &zoneMapBytes,
            &segment.null_bitmap,
            &segment.encoded_data,
        )?;
    }

    writer.finalize(fsync)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CompactionConfig::default();
        assert_eq!(config.min_rows, 100_000);
        assert_eq!(config.max_rows_per_file, 1_000_000);
        assert_eq!(config.max_encoding_threads, 4);
        assert_eq!(config.oltp_p99_threshold_us, 1000);
        assert_eq!(config.check_interval_ms, 5000);
        assert!(config.fsync_enabled);
    }

    #[test]
    fn test_encoding_thread_count() {
        // With 2 columns and max 4 threads
        assert!(encoding_thread_count(2, 4) <= 2);
        // With 100 columns and max 4 threads
        assert!(encoding_thread_count(100, 4) <= 4);
        // With 1 column
        assert_eq!(encoding_thread_count(1, 4), 1);
    }
}

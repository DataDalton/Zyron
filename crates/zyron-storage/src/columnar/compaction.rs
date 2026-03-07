//! Background compaction thread that converts heap rows to .zyr columnar files.
//!
//! Pipeline: materialize rows -> sort by PK -> encode columns (parallel via
//! std::thread::scope) -> build bloom filters -> compute zone maps -> write .zyr file.
//! Runs on a dedicated std::thread with parking_lot::Condvar for wake/sleep.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
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

/// Background compaction thread lifecycle manager.
/// Uses std::thread (no tokio dependency) with Condvar for wakeup.
pub struct CompactionThread {
    config: CompactionConfig,
    thread_handle: Option<JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
    trigger: Arc<(parking_lot::Mutex<bool>, parking_lot::Condvar)>,
}

impl CompactionThread {
    /// Creates and starts the compaction thread.
    pub fn start(config: CompactionConfig) -> Self {
        let stopFlag = Arc::new(AtomicBool::new(false));
        let trigger = Arc::new((parking_lot::Mutex::new(false), parking_lot::Condvar::new()));

        let stopClone = Arc::clone(&stopFlag);
        let triggerClone = Arc::clone(&trigger);
        let cfg = config.clone();

        let handle = std::thread::Builder::new()
            .name("zyron-compaction".into())
            .spawn(move || {
                compaction_loop(cfg, stopClone, triggerClone);
            })
            .expect("failed to spawn compaction thread");

        Self {
            config,
            thread_handle: Some(handle),
            stop_flag: stopFlag,
            trigger,
        }
    }

    /// Triggers an immediate compaction cycle.
    pub fn trigger(&self) {
        let (lock, cvar) = &*self.trigger;
        let mut triggered = lock.lock();
        *triggered = true;
        cvar.notify_one();
    }

    /// Stops the compaction thread and waits for it to finish.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);

        // Wake the thread if it is parked
        let (lock, cvar) = &*self.trigger;
        let mut triggered = lock.lock();
        *triggered = true;
        cvar.notify_one();
        drop(triggered);

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Returns the compaction configuration.
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }
}

impl Drop for CompactionThread {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Main compaction loop. Wakes on trigger or check_interval timeout.
fn compaction_loop(
    config: CompactionConfig,
    stop: Arc<AtomicBool>,
    trigger: Arc<(parking_lot::Mutex<bool>, parking_lot::Condvar)>,
) {
    let checkInterval = std::time::Duration::from_millis(config.check_interval_ms);

    while !stop.load(Ordering::Acquire) {
        // Wait for trigger or timeout
        {
            let (lock, cvar) = &*trigger;
            let mut triggered = lock.lock();
            if !*triggered {
                cvar.wait_for(&mut triggered, checkInterval);
            }
            *triggered = false;
        }

        if stop.load(Ordering::Acquire) {
            break;
        }

        // Compaction cycle placeholder.
        // Full implementation will:
        // 1. Check if enough heap rows have accumulated (>= min_rows)
        // 2. Rate-limit: check OLTP p99, pause 100ms if above threshold
        // 3. Materialize committed heap rows into per-column vectors
        // 4. Sort by primary key
        // 5. Parallel column encoding via std::thread::scope
        // 6. Build bloom filters for high-cardinality columns
        // 7. Compute zone maps
        // 8. Write .zyr file via ZyrFileWriter
        // 9. Log compaction to WAL (CompactionBegin/CompactionEnd)
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

    // Reorder column data by sort permutation.
    // Applies the permutation in-place per column to avoid cloning Vec<u8> per row.
    // Uses the standard cycle-leader algorithm: follow each permutation cycle,
    // moving elements to their final position with O(1) extra space per column.
    let mut reorderedColumns = input.column_data;
    if pkIndex.is_some() {
        for colData in &mut reorderedColumns {
            let mut perm = sortedIndices.clone();
            apply_permutation_in_place(colData, &mut perm);
        }
    }

    // Parallel column encoding via std::thread::scope
    let threadCount = encoding_thread_count(input.columns.len(), config.max_encoding_threads);
    let columns = &input.columns;
    let reordered = &reorderedColumns;

    let segments: Vec<Result<ColumnSegment>> = if threadCount <= 1 || columns.len() <= 1 {
        // Single-threaded path
        columns
            .iter()
            .enumerate()
            .map(|(colIdx, col)| {
                let values: Vec<Option<&[u8]>> =
                    reordered[colIdx].iter().map(|v| v.as_deref()).collect();
                ColumnSegment::build(col.column_id, col.type_id, col.value_size, &values)
            })
            .collect()
    } else {
        // Multi-threaded path
        std::thread::scope(|s| {
            let handles: Vec<_> = columns
                .iter()
                .enumerate()
                .map(|(colIdx, col)| {
                    let colData = &reordered[colIdx];
                    s.spawn(move || {
                        let values: Vec<Option<&[u8]>> =
                            colData.iter().map(|v| v.as_deref()).collect();
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

    // Collect segments, propagating any encoding errors
    let builtSegments: Vec<ColumnSegment> = segments.into_iter().collect::<Result<Vec<_>>>()?;

    // Determine sort order for the file header
    let sortOrder = if pkIndex.is_some() {
        SortOrder::Asc
    } else {
        SortOrder::None
    };

    let pkColumnId = pkIndex.map(|i| input.columns[i].column_id).unwrap_or(0);

    // Generate output file path
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let fileName = format!("table_{}_{}_{}.zyr", input.table_id, rowCount, timestamp);
    let outputPath = config.columnar_dir.join(&fileName);

    // Create output directory if needed
    if let Some(parent) = outputPath.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            zyron_common::ZyronError::CompactionFailed(format!(
                "failed to create columnar directory: {}",
                e
            ))
        })?;
    }

    // Build file header
    let header = ZyrFileHeader {
        format_version: crate::columnar::constants::ZYR_FORMAT_VERSION,
        column_count: input.columns.len() as u32,
        row_count: rowCount as u64,
        table_id: input.table_id,
        xmin_range_lo: input.xmin_lo,
        xmin_range_hi: input.xmin_hi,
        xmax_range_lo: 0,
        xmax_range_hi: 0,
        primary_key_column_id: pkColumnId,
        sort_order: sortOrder,
    };

    // Write .zyr file
    let fileSize = write_zyr_file(&outputPath, header, &builtSegments, config.fsync_enabled)?;

    Ok(CompactionResult {
        file_path: outputPath,
        file_size: fileSize,
        row_count: rowCount as u64,
        column_count: input.columns.len() as u32,
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

    #[test]
    fn test_compaction_thread_start_stop() {
        let config = CompactionConfig {
            check_interval_ms: 50, // Short interval for test
            ..Default::default()
        };

        let mut thread = CompactionThread::start(config);

        // Let it run briefly
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Trigger a cycle
        thread.trigger();

        // Stop
        thread.stop();
        assert!(thread.thread_handle.is_none());
    }

    #[test]
    fn test_compaction_thread_drop_stops() {
        let config = CompactionConfig {
            check_interval_ms: 50,
            ..Default::default()
        };

        // Drop should call stop() via Drop impl
        let _thread = CompactionThread::start(config);
    }
}

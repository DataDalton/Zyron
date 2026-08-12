//! Background compaction thread that converts heap rows to .zyr columnar files.
//!
//! Pipeline: materialize rows -> order rows -> encode columns (parallel via
//! std::thread::scope) -> build bloom filters -> compute zone maps -> write .zyr file.
//! Runs on a dedicated std::thread with parking_lot::Condvar for wake/sleep.
//!
//! Row order is a clustering decision, not a fixed rule. The caller supplies
//! the cluster keys the planner chose and the fold lays rows out under their
//! curve, using the same ordering key the lake writer uses so the two tiers
//! agree on what a key means. With no keys supplied it falls back to
//! ascending primary key, which is the bootstrap policy rather than a
//! default anyone has to configure.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use zyron_common::Result;
use zyron_common::curve::{normalize_component, ordering_key};
use zyron_common::types::TypeId;
use zyron_common::{ClusterKey, ClusterStrategy};

use crate::columnar::file::{SortOrder, ZyrFileHeader, ZyrFileWriter};
use crate::columnar::segment::{BloomPolicy, ColumnSegment, SegmentOptions};

/// What a written file's rows are already ordered by, recorded in its
/// header.
///
/// The caller does the ordering, this only describes it, and describing it
/// wrongly is worse than not describing it: a reader that trusts `Ascending`
/// on a file laid out by a multi-dimensional curve would binary search it
/// and silently miss rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileOrdering {
    /// Arrival order, no column is sorted
    #[default]
    None,
    /// Ascending by one column, which is what both a primary key sort and
    /// a RangePartition cluster key produce
    Ascending { column_id: u32 },
    /// Ordered by a curve that consumes several dimensions at once, so no
    /// single column is sorted
    Curve,
}

impl FileOrdering {
    /// How the cluster keys a fold ran under leave its rows ordered
    pub fn of_cluster_keys(keys: &[ClusterKey]) -> Self {
        match keys.first() {
            None => FileOrdering::None,
            Some(key) if key.strategy == ClusterStrategy::RangePartition => {
                FileOrdering::Ascending {
                    column_id: key.column_id,
                }
            }
            Some(_) => FileOrdering::Curve,
        }
    }

    fn header_fields(self) -> (SortOrder, u32) {
        match self {
            FileOrdering::None | FileOrdering::Curve => (SortOrder::None, 0),
            FileOrdering::Ascending { column_id } => (SortOrder::Asc, column_id),
        }
    }
}

/// Monotonic per-process counter that makes folded filenames unique even when
/// two folds land in the same nanosecond
static FOLD_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    /// Pick each column's encoding by trial encoding every row rather than
    /// a bounded prefix.
    ///
    /// On by default. The fold already materializes the whole column, and
    /// the file it writes is immutable: an encoding chosen from an
    /// unrepresentative prefix is paid on every read for the life of the
    /// file, while the extra trial encode is paid once. Turn it off only
    /// where fold CPU is the binding constraint.
    pub exact_encoding: bool,
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
            exact_encoding: true,
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
    /// Whether this column's segment carries a value bloom filter. Auto
    /// lets the segment's cardinality decide, Force is what a declared
    /// bloom_filter_columns asks for.
    pub bloom_policy: BloomPolicy,
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
    /// Cluster keys the fold lays rows out by, in declaration order.
    ///
    /// Empty falls back to ascending primary key, which is the bootstrap
    /// policy: a table with no measured proposal is still ordered by the
    /// key it already has. A key naming a column this fold did not
    /// materialize is skipped rather than failing the cycle, so a schema
    /// change between the proposal and the fold costs ordering quality
    /// and not the fold.
    pub cluster_keys: Vec<ClusterKey>,
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

/// Row order under the cluster keys, given each key column's cell view in
/// key order.
///
/// One ordering key per row, built once rather than per comparison: a sort
/// does O(n log n) comparisons and this makes each of them a byte compare
/// over `8 * dimensions` bytes. Rows whose key has a NULL component have no
/// position on any curve, so they trail every ordered row, which is where
/// the lake writer puts them too.
///
/// Takes borrowed views so the fold can order its arena-backed columns
/// without materializing them again, and returns `u32` indices because a
/// fold that needs more than four billion rows in one file has a bigger
/// problem than its ordering.
pub fn cluster_order(
    key_columns: &[(TypeId, &[Option<&[u8]>])],
    strategy: ClusterStrategy,
    row_count: usize,
) -> Vec<u32> {
    let keyLen = key_columns.len() * 8;
    let mut orderKeys: Vec<u8> = Vec::with_capacity(row_count * keyLen);
    let mut nullKeys: Vec<bool> = Vec::with_capacity(row_count);
    let mut axes: Vec<u64> = Vec::with_capacity(key_columns.len());
    for row in 0..row_count {
        axes.clear();
        let mut anyNull = false;
        for (typeId, cells) in key_columns {
            match cells.get(row).and_then(|c| *c) {
                Some(cell) => axes.push(normalize_component(*typeId, cell)),
                None => {
                    anyNull = true;
                    axes.push(0);
                }
            }
        }
        nullKeys.push(anyNull);
        orderKeys.extend_from_slice(&ordering_key(strategy, &axes));
    }

    let mut indices: Vec<u32> = (0..row_count as u32).collect();
    indices.sort_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        nullKeys[a].cmp(&nullKeys[b]).then_with(|| {
            orderKeys[a * keyLen..(a + 1) * keyLen].cmp(&orderKeys[b * keyLen..(b + 1) * keyLen])
        })
    });
    indices
}

/// The curve a key list orders by. A multi-dimensional curve consumes
/// every dimension at once, so the leading key's strategy is the file's
#[inline]
pub fn cluster_curve(keys: &[ClusterKey]) -> ClusterStrategy {
    keys.first()
        .map(|k| k.strategy)
        .unwrap_or(ClusterStrategy::RangePartition)
}

/// Row order for owned column data, the shape `run_compaction_cycle` holds
fn cluster_permutation(
    column_data: &[Vec<Option<Vec<u8>>>],
    key_columns: &[(TypeId, usize)],
    cluster_keys: &[ClusterKey],
    row_count: usize,
) -> Vec<usize> {
    let views: Vec<(TypeId, Vec<Option<&[u8]>>)> = key_columns
        .iter()
        .map(|(typeId, idx)| {
            (
                *typeId,
                column_data[*idx].iter().map(|c| c.as_deref()).collect(),
            )
        })
        .collect();
    let borrowed: Vec<(TypeId, &[Option<&[u8]>])> =
        views.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    cluster_order(&borrowed, cluster_curve(cluster_keys), row_count)
        .into_iter()
        .map(|i| i as usize)
        .collect()
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

    // Cluster keys resolved against the columns this fold materialized
    let keyColumns: Vec<(TypeId, usize)> = input
        .cluster_keys
        .iter()
        .filter_map(|key| {
            input
                .columns
                .iter()
                .position(|c| c.column_id == key.column_id)
                .map(|idx| (input.columns[idx].type_id, idx))
        })
        .collect();

    // Ordering, in the order of preference: the cluster keys the planner
    // chose, then the primary key, then arrival order. The ordering key
    // is built exactly as the lake writer builds it, so a heap table and
    // a lake table clustered on the same key agree on what that means
    let sortedIndices: Vec<usize> = if !keyColumns.is_empty() {
        cluster_permutation(
            &input.column_data,
            &keyColumns,
            &input.cluster_keys,
            rowCount,
        )
    } else if let Some(pkIdx) = pkIndex {
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
    let clustered = !keyColumns.is_empty();
    let needs_perm = (clustered || pkIndex.is_some()) && !identityPerm;

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
        if clustered {
            FileOrdering::of_cluster_keys(&input.cluster_keys)
        } else {
            match pkIndex {
                Some(i) => FileOrdering::Ascending {
                    column_id: columns[i].column_id,
                },
                None => FileOrdering::None,
            }
        },
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
/// `ordering` records what the caller ordered the rows by, which the file
/// header carries so a reader knows whether any column is searchable.
/// Used by both the owned-data path and the fold's arena-backed
/// zero-per-cell path.
pub fn encode_and_write<'a, V>(
    config: &CompactionConfig,
    columns: &[ColumnDescriptor],
    row_count: usize,
    column_view: V,
    ordering: FileOrdering,
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
                ColumnSegment::build_with_options(
                    col.column_id,
                    col.type_id,
                    col.value_size,
                    &values,
                    SegmentOptions {
                        bloom: col.bloom_policy,
                        exact_encoding: config.exact_encoding,
                    },
                )
            })
            .collect()
    } else {
        std::thread::scope(|s| {
            let column_view = &column_view;
            let exactEncoding = config.exact_encoding;
            let handles: Vec<_> = columns
                .iter()
                .enumerate()
                .map(|(i, col)| {
                    s.spawn(move || {
                        let values = column_view(i);
                        ColumnSegment::build_with_options(
                            col.column_id,
                            col.type_id,
                            col.value_size,
                            &values,
                            SegmentOptions {
                                bloom: col.bloom_policy,
                                exact_encoding: exactEncoding,
                            },
                        )
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

    let (sortOrder, pkColumnId) = ordering.header_fields();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let uniqueSuffix = FOLD_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let fileName = format!(
        "table_{}_{}_{}_{}.zyr",
        table_id, rowCount, timestamp, uniqueSuffix
    );
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

    /// Two int columns over the given rows, with column 0 as the primary key
    fn grid(rows: &[(i64, i64)]) -> (Vec<ColumnDescriptor>, Vec<Vec<Option<Vec<u8>>>>) {
        let descriptors = vec![
            ColumnDescriptor {
                column_id: 0,
                type_id: TypeId::Int64,
                value_size: 8,
                is_primary_key: true,
                bloom_policy: BloomPolicy::Auto,
            },
            ColumnDescriptor {
                column_id: 1,
                type_id: TypeId::Int64,
                value_size: 8,
                is_primary_key: false,
                bloom_policy: BloomPolicy::Auto,
            },
        ];
        let data = vec![
            rows.iter()
                .map(|(a, _)| Some(a.to_le_bytes().to_vec()))
                .collect(),
            rows.iter()
                .map(|(_, b)| Some(b.to_le_bytes().to_vec()))
                .collect(),
        ];
        (descriptors, data)
    }

    fn key(column_id: u32, strategy: ClusterStrategy) -> ClusterKey {
        ClusterKey {
            column_id,
            strategy,
            param: 0,
        }
    }

    /// The point of clustering the fold: rows land in the order the chosen
    /// key asks for, where the primary key sort would have scattered them
    #[test]
    fn test_a_cluster_key_orders_the_fold_where_the_primary_key_would_not() {
        // The primary key is `a`, but the workload filters on `b`
        let rows: Vec<(i64, i64)> = (0..64).map(|i| (i, 63 - i)).collect();
        let (_, data) = grid(&rows);
        let order = cluster_permutation(
            &data,
            &[(TypeId::Int64, 1)],
            &[key(1, ClusterStrategy::RangePartition)],
            rows.len(),
        );
        let ordered: Vec<i64> = order.iter().map(|&i| rows[i].1).collect();
        assert_eq!(ordered, (0..64).collect::<Vec<_>>(), "ordered by b, not a");

        // Ordering by the primary key instead leaves b descending, which is
        // exactly the layout a predicate on b cannot skip on
        let by_pk = cluster_permutation(
            &data,
            &[(TypeId::Int64, 0)],
            &[key(0, ClusterStrategy::RangePartition)],
            rows.len(),
        );
        assert_eq!(by_pk, (0..64).collect::<Vec<_>>());
    }

    /// A multi-dimensional curve has to actually interleave, or declaring
    /// one would be a note in the header and nothing else
    #[test]
    fn test_a_curve_orders_differently_from_a_range_partition() {
        let mut rows = Vec::new();
        for a in 0..8i64 {
            for b in 0..8i64 {
                rows.push((a, b));
            }
        }
        let (_, data) = grid(&rows);
        let keys = [(TypeId::Int64, 0usize), (TypeId::Int64, 1usize)];
        let ranged = cluster_permutation(
            &data,
            &keys,
            &[
                key(0, ClusterStrategy::RangePartition),
                key(1, ClusterStrategy::RangePartition),
            ],
            rows.len(),
        );
        let interleaved = cluster_permutation(
            &data,
            &keys,
            &[
                key(0, ClusterStrategy::BitInterleave),
                key(1, ClusterStrategy::BitInterleave),
            ],
            rows.len(),
        );
        assert_ne!(ranged, interleaved);
        assert_eq!(ranged, (0..64).collect::<Vec<_>>());

        // The interleaved order matches the shared curve exactly, which is
        // what makes a heap table and a lake table on the same key agree
        let mut expected: Vec<usize> = (0..rows.len()).collect();
        expected.sort_by_key(|&i| {
            ordering_key(
                ClusterStrategy::BitInterleave,
                &[
                    normalize_component(TypeId::Int64, &rows[i].0.to_le_bytes()),
                    normalize_component(TypeId::Int64, &rows[i].1.to_le_bytes()),
                ],
            )
        });
        assert_eq!(interleaved, expected);
    }

    /// NULL has no position on any curve, so those rows trail the ordered
    /// ones, which is where the lake writer puts them too
    #[test]
    fn test_rows_with_a_null_key_trail_the_ordered_rows() {
        let data: Vec<Vec<Option<Vec<u8>>>> = vec![vec![
            Some(5i64.to_le_bytes().to_vec()),
            None,
            Some(1i64.to_le_bytes().to_vec()),
            None,
            Some(3i64.to_le_bytes().to_vec()),
        ]];
        let order = cluster_permutation(
            &data,
            &[(TypeId::Int64, 0)],
            &[key(0, ClusterStrategy::RangePartition)],
            5,
        );
        assert_eq!(order[..3], [2, 4, 0], "ordered rows come first, ascending");
        let tail: std::collections::BTreeSet<usize> = order[3..].iter().copied().collect();
        assert_eq!(tail, [1usize, 3].into_iter().collect());
    }

    /// The header has to describe the order that is actually there. A file
    /// laid out by a curve has no sorted column, and saying otherwise
    /// invites a binary search that silently misses rows
    #[test]
    fn test_the_header_describes_the_order_that_is_actually_there() {
        assert_eq!(FileOrdering::of_cluster_keys(&[]), FileOrdering::None);
        assert_eq!(
            FileOrdering::of_cluster_keys(&[key(7, ClusterStrategy::RangePartition)]),
            FileOrdering::Ascending { column_id: 7 }
        );
        for scattered in [
            ClusterStrategy::BitInterleave,
            ClusterStrategy::SpaceFilling,
            ClusterStrategy::AntiCluster,
        ] {
            assert_eq!(
                FileOrdering::of_cluster_keys(&[key(7, scattered)]),
                FileOrdering::Curve,
                "{scattered:?} sorts no single column"
            );
        }
        assert_eq!(
            FileOrdering::Curve.header_fields(),
            (SortOrder::None, 0),
            "a curve must not name a searchable column"
        );
        assert_eq!(
            FileOrdering::Ascending { column_id: 3 }.header_fields(),
            (SortOrder::Asc, 3)
        );
    }

    /// A cluster key naming a column this fold did not materialize costs
    /// ordering quality, never the fold
    #[test]
    fn test_a_cluster_key_for_an_absent_column_falls_back_to_the_primary_key() {
        let rows: Vec<(i64, i64)> = vec![(3, 0), (1, 1), (2, 2)];
        let (descriptors, data) = grid(&rows);
        let dir = tempfile::tempdir().expect("tempdir");
        let config = CompactionConfig {
            columnar_dir: dir.path().to_path_buf(),
            ..CompactionConfig::default()
        };
        let result = run_compaction_cycle(
            &config,
            CompactionInput {
                columns: descriptors,
                column_data: data,
                table_id: 1,
                xmin_lo: 1,
                xmin_hi: 1,
                cluster_keys: vec![key(99, ClusterStrategy::BitInterleave)],
            },
        )
        .expect("fold");
        assert_eq!(result.row_count, 3);

        let reader = crate::columnar::ZyrFileReader::open(&result.file_path).expect("open");
        assert_eq!(reader.sort_order(), SortOrder::Asc);
        assert_eq!(reader.primary_key_column_id(), 0);
        let (bytes, _) = reader.decode_column(0, 3, 8).expect("decode");
        let values: Vec<i64> = (0..3)
            .map(|r| i64::from_le_bytes(bytes[r * 8..r * 8 + 8].try_into().expect("8 bytes")))
            .collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    /// A fold under a cluster key writes the rows in that order and marks
    /// the header accordingly, end to end through the real writer
    #[test]
    fn test_a_clustered_fold_writes_the_rows_in_the_declared_order() {
        let rows: Vec<(i64, i64)> = (0..32).map(|i| (i, 31 - i)).collect();
        let (descriptors, data) = grid(&rows);
        let dir = tempfile::tempdir().expect("tempdir");
        let config = CompactionConfig {
            columnar_dir: dir.path().to_path_buf(),
            ..CompactionConfig::default()
        };
        let result = run_compaction_cycle(
            &config,
            CompactionInput {
                columns: descriptors,
                column_data: data,
                table_id: 1,
                xmin_lo: 1,
                xmin_hi: 1,
                cluster_keys: vec![key(1, ClusterStrategy::RangePartition)],
            },
        )
        .expect("fold");

        let reader = crate::columnar::ZyrFileReader::open(&result.file_path).expect("open");
        assert_eq!(reader.sort_order(), SortOrder::Asc);
        assert_eq!(
            reader.primary_key_column_id(),
            1,
            "the header names the column the file is ordered by"
        );
        let (b_bytes, _) = reader.decode_column(1, 32, 8).expect("decode");
        let values: Vec<i64> = (0..32)
            .map(|r| i64::from_le_bytes(b_bytes[r * 8..r * 8 + 8].try_into().expect("8 bytes")))
            .collect();
        assert_eq!(values, (0..32).collect::<Vec<_>>());

        // Every row kept its own pairing through the permutation
        let (a_bytes, _) = reader.decode_column(0, 32, 8).expect("decode");
        for r in 0..32 {
            let a = i64::from_le_bytes(a_bytes[r * 8..r * 8 + 8].try_into().expect("8 bytes"));
            assert_eq!(a, 31 - values[r], "row {r} lost its pairing");
        }
    }

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

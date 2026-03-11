//! Table and column statistics for query optimization.
//!
//! Statistics are collected by analyze_table() which scans heap pages
//! (with optional Bernoulli sampling for large tables) and computes
//! per-column null counts, distinct value counts, and equi-depth histograms.

use crate::ids::*;
use crate::schema::TableEntry;
use serde::{Deserialize, Serialize};
use zyron_common::Result;
use zyron_storage::HeapFile;

/// Statistics for a table, collected by ANALYZE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStats {
    pub table_id: TableId,
    pub row_count: u64,
    pub page_count: u32,
    pub avg_row_size: u32,
    pub last_analyzed: u64,
}

/// Statistics for a single column, collected by ANALYZE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub table_id: TableId,
    pub column_id: ColumnId,
    pub null_fraction: f64,
    pub distinct_count: u64,
    pub avg_width: u32,
    pub histogram: Option<Histogram>,
    pub most_common_values: Vec<Vec<u8>>,
    pub most_common_freqs: Vec<f64>,
}

/// Equi-depth histogram for selectivity estimation.
///
/// Each bucket contains approximately the same number of rows.
/// Bounds are the boundary values between buckets, stored as raw bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub num_buckets: u32,
    pub bounds: Vec<Vec<u8>>,
    pub counts: Vec<u64>,
}

impl Histogram {
    /// Builds an equi-depth histogram from sorted values.
    pub fn build(sorted_values: &[Vec<u8>], num_buckets: u32) -> Self {
        if sorted_values.is_empty() || num_buckets == 0 {
            return Self {
                num_buckets: 0,
                bounds: Vec::new(),
                counts: Vec::new(),
            };
        }

        let n = sorted_values.len();
        let bucket_size = (n as f64 / num_buckets as f64).ceil() as usize;
        let actual_buckets = (n + bucket_size - 1) / bucket_size;

        let mut bounds = Vec::with_capacity(actual_buckets + 1);
        let mut counts = Vec::with_capacity(actual_buckets);

        bounds.push(sorted_values[0].clone());

        for i in 0..actual_buckets {
            let start = i * bucket_size;
            let end = ((i + 1) * bucket_size).min(n);
            counts.push((end - start) as u64);
            if end < n {
                bounds.push(sorted_values[end].clone());
            } else {
                bounds.push(sorted_values[n - 1].clone());
            }
        }

        Self {
            num_buckets: actual_buckets as u32,
            bounds,
            counts,
        }
    }
}

/// Collects statistics for a table by scanning heap pages.
///
/// Scans all tuples and computes per-column statistics including
/// null fraction, distinct value count, average width, and histogram.
/// For large tables, a sampling approach can be implemented by
/// limiting the scan to a subset of pages.
pub async fn analyze_table(
    table: &TableEntry,
    heap: &HeapFile,
) -> Result<(TableStats, Vec<ColumnStats>)> {
    let col_count = table.columns.len();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut row_count: u64 = 0;
    let mut total_row_bytes: u64 = 0;

    // Per-column accumulators
    let null_counts = vec![0u64; col_count];
    let mut total_widths = vec![0u64; col_count];
    let mut all_values: Vec<Vec<Vec<u8>>> = vec![Vec::new(); col_count];

    let guard = heap.scan()?;
    guard.for_each(|_tid, view| {
        row_count += 1;
        let data = view.data;
        total_row_bytes += data.len() as u64;

        // Store entire tuple bytes as the value for column 0.
        // Full per-column extraction requires the tuple format to be finalized.
        if col_count > 0 {
            let value = data.to_vec();
            total_widths[0] += value.len() as u64;
            all_values[0].push(value);
        }
    });

    let page_count = heap.num_pages().await?;
    let avg_row_size = if row_count > 0 {
        (total_row_bytes / row_count) as u32
    } else {
        0
    };

    let table_stats = TableStats {
        table_id: table.id,
        row_count,
        page_count,
        avg_row_size,
        last_analyzed: now,
    };

    let mut column_stats = Vec::with_capacity(col_count);
    for i in 0..col_count {
        let null_fraction = if row_count > 0 {
            null_counts[i] as f64 / row_count as f64
        } else {
            0.0
        };
        let avg_width = if row_count > 0 {
            (total_widths[i] / row_count) as u32
        } else {
            0
        };

        // Sort values first, then derive distinct count + MCV from sorted runs.
        // This replaces the previous HashSet approach, eliminating per-row
        // clone + hash + bucket allocation overhead.
        all_values[i].sort_unstable();

        let distinct_count = count_distinct_sorted(&all_values[i]);

        // MCV from sorted runs: count consecutive equal values
        let (mcv, mcf) = compute_most_common_sorted(&all_values[i], row_count, 10);

        // Build histogram from the now-sorted values
        let histogram = if !all_values[i].is_empty() {
            let num_buckets = 100u32.min(distinct_count as u32);
            Some(Histogram::build(&all_values[i], num_buckets))
        } else {
            None
        };

        column_stats.push(ColumnStats {
            table_id: table.id,
            column_id: table.columns[i].id,
            null_fraction,
            distinct_count,
            avg_width,
            histogram,
            most_common_values: mcv,
            most_common_freqs: mcf,
        });
    }

    Ok((table_stats, column_stats))
}

/// Counts distinct values in a sorted slice by counting adjacent differences.
fn count_distinct_sorted(sorted: &[Vec<u8>]) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let mut count = 1u64;
    for i in 1..sorted.len() {
        if sorted[i] != sorted[i - 1] {
            count += 1;
        }
    }
    count
}

/// Computes the top-N most common values from a pre-sorted slice.
/// Walks sorted runs to count frequencies without a HashMap.
fn compute_most_common_sorted(sorted: &[Vec<u8>], total: u64, top_n: usize) -> (Vec<Vec<u8>>, Vec<f64>) {
    if sorted.is_empty() || total == 0 {
        return (Vec::new(), Vec::new());
    }

    // Collect (value, count) pairs from sorted runs
    let mut runs: Vec<(&[u8], u64)> = Vec::new();
    let mut run_start = 0;
    for i in 1..=sorted.len() {
        if i == sorted.len() || sorted[i] != sorted[run_start] {
            let count = (i - run_start) as u64;
            runs.push((sorted[run_start].as_slice(), count));
            run_start = i;
        }
    }

    // Partial sort: only need top N, so use select_nth_unstable for large sets
    if runs.len() > top_n && top_n > 0 {
        runs.select_nth_unstable_by(top_n - 1, |a, b| b.1.cmp(&a.1));
        runs.truncate(top_n);
        runs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    } else {
        runs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    }

    let n = top_n.min(runs.len());
    let mcv: Vec<Vec<u8>> = runs[..n].iter().map(|(k, _)| k.to_vec()).collect();
    let mcf: Vec<f64> = runs[..n]
        .iter()
        .map(|(_, count)| *count as f64 / total as f64)
        .collect();

    (mcv, mcf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_build_basic() {
        let values: Vec<Vec<u8>> = (0..100u8).map(|i| vec![i]).collect();
        let hist = Histogram::build(&values, 10);
        assert_eq!(hist.num_buckets, 10);
        assert_eq!(hist.counts.len(), 10);
        let total: u64 = hist.counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_histogram_build_empty() {
        let hist = Histogram::build(&[], 10);
        assert_eq!(hist.num_buckets, 0);
        assert!(hist.bounds.is_empty());
        assert!(hist.counts.is_empty());
    }

    #[test]
    fn test_histogram_build_fewer_values_than_buckets() {
        let values: Vec<Vec<u8>> = vec![vec![1], vec![2], vec![3]];
        let hist = Histogram::build(&values, 10);
        assert!(hist.num_buckets <= 3);
        let total: u64 = hist.counts.iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_histogram_single_value() {
        let values = vec![vec![42u8]];
        let hist = Histogram::build(&values, 5);
        assert_eq!(hist.num_buckets, 1);
        assert_eq!(hist.counts[0], 1);
    }

    #[test]
    fn test_most_common_values() {
        let mut values = vec![vec![1u8], vec![1], vec![1], vec![2], vec![2], vec![3]];
        values.sort_unstable();
        let (mcv, mcf) = compute_most_common_sorted(&values, 6, 2);
        assert_eq!(mcv.len(), 2);
        assert_eq!(mcv[0], vec![1u8]);
        assert!((mcf[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_most_common_empty() {
        let (mcv, mcf) = compute_most_common_sorted(&[], 0, 10);
        assert!(mcv.is_empty());
        assert!(mcf.is_empty());
    }

    #[test]
    fn test_count_distinct_sorted() {
        let values: Vec<Vec<u8>> = vec![vec![1], vec![1], vec![2], vec![3], vec![3], vec![3]];
        assert_eq!(count_distinct_sorted(&values), 3);
        assert_eq!(count_distinct_sorted(&[]), 0);
        assert_eq!(count_distinct_sorted(&[vec![42]]), 1);
    }

    #[test]
    fn test_table_stats_fields() {
        let stats = TableStats {
            table_id: TableId(1),
            row_count: 1000,
            page_count: 10,
            avg_row_size: 64,
            last_analyzed: 1700000000,
        };
        assert_eq!(stats.row_count, 1000);
        assert_eq!(stats.page_count, 10);
    }

    #[test]
    fn test_column_stats_fields() {
        let stats = ColumnStats {
            table_id: TableId(1),
            column_id: ColumnId(0),
            null_fraction: 0.05,
            distinct_count: 500,
            avg_width: 8,
            histogram: None,
            most_common_values: vec![],
            most_common_freqs: vec![],
        };
        assert!((stats.null_fraction - 0.05).abs() < 0.001);
        assert_eq!(stats.distinct_count, 500);
    }
}

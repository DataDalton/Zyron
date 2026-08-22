//! Manifest statistics for the planner.
//!
//! A lake table never runs ANALYZE. Its writer computes exact per-column
//! bounds, null counts and row counts for every file it produces, and the
//! manifest is the accumulated result, so the statistics a plan needs are
//! already durable by the time the commit lands. Publishing them into the
//! catalog's statistics cache after every commit is what keeps a lake table
//! off the planner's no-statistics defaults.
//!
//! Row counts are per file and predicate deletes are not applied row by row,
//! so a file carrying an active delete predicate still reports the rows it
//! was written with. That overstates live rows until an optimize rewrites
//! the file, which is the same direction ANALYZE errs in between runs.

use zyron_catalog::stats::{ColumnStats, TableStats};
use zyron_catalog::{Catalog, TableEntry};
use zyron_common::page::PAGE_SIZE;
use zyron_lake::ManifestFile;

/// Derives table and column statistics from one manifest version.
pub fn manifest_stats(
    table: &TableEntry,
    manifest: &ManifestFile,
) -> (TableStats, Vec<ColumnStats>) {
    let mut row_count = 0u64;
    let mut byte_count = 0u64;
    for entry in &manifest.entries {
        row_count += entry.row_count;
        byte_count += entry.size_bytes;
    }

    let table_stats = TableStats {
        table_id: table.id,
        row_count,
        page_count: byte_count.div_ceil(PAGE_SIZE as u64).min(u32::MAX as u64) as u32,
        avg_row_size: if row_count == 0 {
            0
        } else {
            (byte_count / row_count).min(u32::MAX as u64) as u32
        },
        // The version's own timestamp: these statistics are exactly as fresh
        // as the commit that produced them
        last_analyzed: (manifest.timestamp_us / 1_000_000).max(0) as u64,
    };

    let mut column_stats = Vec::with_capacity(table.columns.len());
    for column in &table.columns {
        let column_id = column.id.0 as u32;
        let mut nulls = 0u64;
        let mut rows_with_stats = 0u64;
        for entry in &manifest.entries {
            if let Some(stats) = entry.stats_for(column_id) {
                nulls += stats.bounds.null_count;
                rows_with_stats += stats.bounds.row_count;
            }
        }
        column_stats.push(ColumnStats {
            table_id: table.id,
            column_id: column.id,
            null_fraction: if rows_with_stats == 0 {
                0.0
            } else {
                nulls as f64 / rows_with_stats as f64
            },
            // The manifest carries exact bounds and null counts but no
            // distinct count, so nothing is claimed here rather than a
            // guess the selectivity estimator would treat as measured
            distinct_count: 0,
            avg_width: column
                .type_id
                .fixed_size()
                .map(|n| n as u32)
                .unwrap_or_else(|| {
                    if table_stats.row_count == 0 || table.columns.is_empty() {
                        0
                    } else {
                        table_stats.avg_row_size / table.columns.len() as u32
                    }
                }),
            histogram: None,
            most_common_values: Vec::new(),
            most_common_freqs: Vec::new(),
        });
    }

    (table_stats, column_stats)
}

/// Publishes a manifest's statistics into the catalog cache, replacing what
/// an earlier version left. Memory only, the manifest on disk is the durable
/// copy, so a restart reloads them from the recovered log.
pub fn publish_manifest_stats(catalog: &Catalog, table: &TableEntry, manifest: &ManifestFile) {
    let (table_stats, column_stats) = manifest_stats(table, manifest);
    catalog.put_stats(table.id, table_stats, column_stats);
}

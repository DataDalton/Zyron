//! Data file writer.
//!
//! Turns one sorted row batch into one immutable .zyr file at the lake's
//! partition-addressed path and returns the manifest entry describing it.
//! Lake files carry only user columns, visibility is by log version, so
//! there are no MVCC system columns and no patch store.
//!
//! Statistics are exact. Min, max and null count come off the segment
//! build, which orders and counts the column to lay its zone maps and
//! bloom down, so the manifest entry is filled from a pass the write was
//! making anyway rather than a second one. A declared bloom column's filter
//! is copied into the entry as well, so pruning probes it with zero IO,
//! while an undeclared column's filter stays in the segment where it costs
//! the manifest nothing

use std::path::Path;

use zyron_common::{TypeId, ZyronError};
use zyron_storage::columnar::{
    BloomPolicy, ColumnSegment, SegmentOptions, SortOrder, ZyrFileHeader, ZyrFileWriter,
};

use crate::cells::{CellFamily, cell_family, cell_to_value, compare_cells};
use crate::curve::{normalize_component, ordering_key_into};
use crate::manifest::ClusterStrategy;
use crate::manifest::{ColumnStatsEntry, PartitionEntry};
use crate::paths::{LakePaths, data_file_name, index_file_name};
use crate::predicate::ColumnBounds;
use crate::schema::LakeSchema;

/// One column of a row batch, cells in row order, None is NULL
#[derive(Debug, Clone)]
pub struct ColumnData {
    pub column_id: u32,
    pub cells: Vec<Option<Vec<u8>>>,
}

/// One file's worth of rows plus the layout it is written under
#[derive(Debug)]
pub struct WriteRequest<'a> {
    /// Data file identity, must be unallocated in the target manifest
    pub partition_id: u64,
    /// Every schema column exactly once, any order
    pub columns: &'a [ColumnData],
    /// Column ids to sort by, in declaration order, nulls last. Empty
    /// writes rows in arrival order
    pub sort_keys: &'a [u32],
    /// Ordering curve each key column asks for, parallel to `sort_keys`.
    /// A shorter list leaves the remaining keys on RangePartition, which is
    /// plain ascending order
    pub sort_strategies: &'a [ClusterStrategy],
    /// Cluster spec the layout was chosen by, recorded per file so a past
    /// version reads its own layout metadata
    pub cluster_spec_id: u32,
    pub table_id: u64,
    /// Columns a table property declared a bloom filter for, forced whatever
    /// their cardinality says. Every other column keeps the cardinality
    /// heuristic
    pub bloom_columns: &'a [u32],
    /// The index this file belongs to, which decides its name. None writes
    /// a table data file, Some writes one file of that index. Both kinds
    /// sit in `data/` and both are `.zyr`, so the name is what separates
    /// them and a sweep never has to open a file to classify it
    pub index_id: Option<u32>,
}

impl WriteRequest<'_> {
    /// The name this request's file takes, wherever it is written
    fn file_name(&self) -> String {
        match self.index_id {
            None => data_file_name(self.partition_id),
            Some(index_id) => index_file_name(index_id, self.partition_id),
        }
    }
}

/// One written file and the row order it was written in.
///
/// `order[ordinal]` is the input row that landed at that ordinal. Index
/// maintenance needs it to address rows the write just placed, and having
/// it here is what keeps an index delta from re-reading a file the writer
/// has this instant finished producing
#[derive(Debug)]
pub struct WrittenFile {
    pub entry: PartitionEntry,
    pub order: Vec<usize>,
}

/// Writes one data file into the table's `data/` directory and returns
/// its manifest entry. The file appears atomically, the storage writer
/// stages to a temp name and renames on finalize. Nothing references the
/// file until a commit adds the entry
pub fn write_data_file(
    paths: &LakePaths,
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
) -> Result<PartitionEntry, ZyronError> {
    write_data_file_at(&paths.data_dir(), schema, req).map(|w| w.entry)
}

/// Writes one data file and keeps the row order it produced, for a caller
/// that has to address the rows it just wrote
pub fn write_data_file_ordered(
    paths: &LakePaths,
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
) -> Result<WrittenFile, ZyronError> {
    write_data_file_at(&paths.data_dir(), schema, req)
}

/// Writes one data file into an arbitrary directory under its partition
/// name. A clustering pass writes candidates into `_staging/pass_<id>/`
/// and renames them into `data/` only once the gate accepts them, which
/// is what keeps a rejected pass from ever touching the active set
pub fn write_data_file_at(
    dir: &Path,
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
) -> Result<WrittenFile, ZyronError> {
    let row_count = validate_batch(schema, req.columns)?;
    let order = stored_order(schema, req, row_count)?;
    // The curve the leading key declared, which decides whether the header
    // may claim the rows ascend by it
    let leading_strategy = req
        .sort_strategies
        .first()
        .copied()
        .unwrap_or(ClusterStrategy::RangePartition);

    std::fs::create_dir_all(dir)?;
    let path = dir.join(req.file_name());
    if path.exists() {
        return Err(ZyronError::Internal(format!(
            "data file for partition {:#x} already exists at {}",
            req.partition_id,
            path.display()
        )));
    }
    let header = ZyrFileHeader {
        format_version: zyron_storage::columnar::ZYR_FORMAT_VERSION,
        column_count: schema.columns.len() as u32,
        row_count: row_count as u64,
        table_id: req.table_id,
        xmin_range_lo: 0,
        xmin_range_hi: 0,
        xmax_range_lo: 0,
        xmax_range_hi: 0,
        primary_key_column_id: req.sort_keys.first().copied().unwrap_or(0),
        // Only a range partition orders rows ascending by the leading key.
        // The other curves consume every key at once, so rows land in curve
        // order and the leading column is not monotonic across the file. A
        // header claiming Asc there would let an ordered lookup binary
        // search a column that is not sorted and miss rows that are present
        sort_order: match (req.sort_keys.is_empty(), leading_strategy) {
            (true, _) => SortOrder::None,
            (false, ClusterStrategy::RangePartition) => SortOrder::Asc,
            (false, _) => SortOrder::None,
        },
        // Filled in by finalize once the index has a position
        segment_index_offset: 0,
        segment_index_size: 0,
    };
    let mut writer = ZyrFileWriter::create(&path, header)?;

    // Columns are written and described in schema order so the manifest
    // stats arrive sorted by column id after the id sort below
    let mut column_stats = Vec::with_capacity(schema.columns.len());
    for col in &schema.columns {
        let data = req
            .columns
            .iter()
            .find(|c| c.column_id == col.id)
            .ok_or_else(|| ZyronError::Internal(format!("column \"{}\" has no data", col.name)))?;
        let physical = col.physical_type_id();
        let value_size = physical.fixed_size().unwrap_or(0);

        // The column in stored order. Everything the manifest records about
        // it comes back off the segment build, which walks these same values
        // to pack and encode them, so this loop places rows and nothing else
        let mut views: Vec<Option<&[u8]>> = Vec::with_capacity(row_count);
        for &row in &order {
            views.push(data.cells[row].as_deref());
        }

        // The whole column is materialized here already, so the encoding
        // is picked by trial encoding every row rather than a prefix, and
        // the distinct count is the whole column's rather than a capped one
        let declared_bloom = req.bloom_columns.contains(&col.id);
        let options = SegmentOptions {
            bloom: if declared_bloom {
                BloomPolicy::Force
            } else {
                BloomPolicy::Auto
            },
            exact_encoding: true,
            distinct_sketch: true,
        };
        let segment =
            ColumnSegment::build_with_options(col.id, physical, value_size, &views, options)?;
        let (zone_bytes, bloom_bytes) = segment_frame_bytes(&segment);
        let segment_bytes = writer.write_segment(
            col.id,
            &segment.header.to_bytes(),
            bloom_bytes.as_deref(),
            &zone_bytes,
            &segment.null_bitmap,
            &segment.encoded_data,
        )?;

        // Only a declared column's filter is carried into the manifest. A
        // bloom is ten bits per value, so carrying every column's would make
        // the manifest grow with the rows in the table rather than with the
        // files in it, and the manifest is held in memory per version. Every
        // filter is still written into the segment, where a probe that has
        // opened the file reads it from the offset the header records. What
        // a declared column buys is the probe that answers before the file
        // is opened at all
        let bloom_for_manifest = if declared_bloom { bloom_bytes } else { None };
        column_stats.push(column_stats_entry(
            col,
            &views,
            &segment,
            row_count,
            bloom_for_manifest,
            segment_bytes,
        ));
    }
    let size_bytes = writer.finalize(true)?;

    column_stats.sort_by_key(|s| s.column_id);
    Ok(WrittenFile {
        entry: PartitionEntry {
            partition_id: req.partition_id,
            size_bytes,
            row_count: row_count as u64,
            added_version: 0,
            cluster_spec_id: req.cluster_spec_id,
            column_stats: std::sync::Arc::new(column_stats),
            delete_predicate_ids: Vec::new(),
        },
        order,
    })
}

/// The order this batch's rows are stored in: a permutation over row
/// indices ordered by the key columns under the curve each one declared,
/// nulls last, stable for determinism. A batch with no sort key stores rows
/// in arrival order, and a declared strategy that did not change the order
/// would be a note in the manifest and nothing else.
///
/// The curve is the first key's, because a multi-dimensional curve consumes
/// every dimension at once and keys declared after it only refine rows the
/// curve placed together. It decides both the permutation and whether the
/// header may claim ascending order.
///
/// This is the whole of what a cluster key costs a write, which is why it
/// is reachable on its own: measuring it as the difference between a keyed
/// and an unkeyed file makes it the gap between two numbers that carry the
/// encode and the fsync, and a pass costing tens of microseconds does not
/// survive that subtraction
pub fn stored_order(
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
    row_count: usize,
) -> Result<Vec<usize>, ZyronError> {
    if req.sort_keys.is_empty() {
        return Ok((0..row_count).collect());
    }
    let leading_strategy = req
        .sort_strategies
        .first()
        .copied()
        .unwrap_or(ClusterStrategy::RangePartition);

    let mut keys = Vec::with_capacity(req.sort_keys.len());
    for key_id in req.sort_keys {
        let col = schema.column_by_id(*key_id).ok_or_else(|| {
            ZyronError::Internal(format!("sort key column {} is not in the schema", key_id))
        })?;
        let data = req
            .columns
            .iter()
            .find(|c| c.column_id == *key_id)
            .ok_or_else(|| {
                ZyronError::Internal(format!("sort key column {} has no data", key_id))
            })?;
        keys.push((col.physical_type_id(), data));
    }
    // One ordering key per row, built once rather than per comparison: a
    // sort does O(n log n) comparisons and this makes each of them a byte
    // compare over `8 * dimensions` bytes.
    //
    // The keys are packed end to end in one buffer at a fixed stride, the
    // same shape the clustering pass carries them in. A key per row in its
    // own `Vec` costs an allocation per row and leaves every one of those
    // comparisons dereferencing a pointer to reach the bytes, which on a
    // million row file is a million allocations under a sort that touches
    // them twenty million times
    let key_len = keys.len() * 8;
    let mut key_bytes = vec![0u8; row_count * key_len];
    let mut null_key = vec![false; row_count];
    let mut axes = Vec::with_capacity(keys.len());
    for row in 0..row_count {
        axes.clear();
        let mut any_null = false;
        for (physical, data) in &keys {
            match &data.cells[row] {
                Some(cell) => axes.push(normalize_component(*physical, cell)),
                None => {
                    any_null = true;
                    axes.push(0);
                }
            }
        }
        // Nulls last, whatever the curve: they have no position on it
        null_key[row] = any_null;
        ordering_key_into(
            leading_strategy,
            &axes,
            &mut key_bytes[row * key_len..(row + 1) * key_len],
        );
    }

    let key_at = |row: usize| &key_bytes[row * key_len..(row + 1) * key_len];
    let mut order: Vec<usize> = (0..row_count).collect();
    order.sort_by(|&a, &b| {
        null_key[a]
            .cmp(&null_key[b])
            .then_with(|| key_at(a).cmp(key_at(b)))
    });
    Ok(order)
}

/// The column's smallest and largest stored cell, at full width.
///
/// A segment build already ordered the column to lay its zone maps down, so
/// the rows it landed on are the rows the manifest bounds come from, and no
/// second comparison pass is needed. A fixed-width byte family is the
/// exception: every lake reader orders those cells by their bytes from the
/// first, and a stat slot orders them as a little endian integer, so a
/// column of that shape is compared here. It is the same shape the stored
/// filter refuses to push down for
/// The two byte regions the file writer is handed alongside a built
/// segment: its zone maps flattened, and its bloom filter serialized.
///
/// Reachable on its own because it is real per column work that is neither
/// encoding nor IO, and a phase measured as the gap between a whole write
/// and the parts of it that were timed is a remainder carrying every other
/// measurement's error
pub fn segment_frame_bytes(segment: &ColumnSegment) -> (Vec<u8>, Option<Vec<u8>>) {
    let zone_bytes: Vec<u8> = segment
        .zone_maps
        .iter()
        .flat_map(|z| z.to_bytes())
        .collect();
    let bloom_bytes = segment.bloom_filter.as_ref().map(|b| b.to_bytes());
    (zone_bytes, bloom_bytes)
}

/// The manifest entry describing one written column.
///
/// Bounds come off the segment build's own extremes rather than a second
/// walk of the column, except where the segment's slot order is not the
/// order the lake compares those cells in. `bloom_for_manifest` is the
/// filter to carry, which is None for a column that did not declare one
pub fn column_stats_entry(
    col: &crate::schema::LakeColumn,
    views: &[Option<&[u8]>],
    segment: &ColumnSegment,
    row_count: usize,
    bloom_for_manifest: Option<Vec<u8>>,
    segment_bytes: u64,
) -> ColumnStatsEntry {
    let physical = col.physical_type_id();
    let (min_cell, max_cell) = column_extrema(physical, views, segment);
    ColumnStatsEntry {
        column_id: col.id,
        bounds: ColumnBounds {
            min: min_cell.and_then(|c| cell_to_value(physical, c)),
            max: max_cell.and_then(|c| cell_to_value(physical, c)),
            null_count: segment.header.null_count,
            row_count: row_count as u64,
        },
        bloom: bloom_for_manifest,
        ndv: segment.ndv,
        size_bytes: Some(segment_bytes),
    }
}

fn column_extrema<'a>(
    physical: TypeId,
    views: &[Option<&'a [u8]>],
    segment: &ColumnSegment,
) -> (Option<&'a [u8]>, Option<&'a [u8]>) {
    if physical.fixed_size().is_some() && cell_family(physical) == CellFamily::Bytes {
        let mut min_cell: Option<&'a [u8]> = None;
        let mut max_cell: Option<&'a [u8]> = None;
        for cell in views.iter().flatten().copied() {
            if min_cell.is_none_or(|m| compare_cells(physical, cell, m).is_lt()) {
                min_cell = Some(cell);
            }
            if max_cell.is_none_or(|m| compare_cells(physical, cell, m).is_gt()) {
                max_cell = Some(cell);
            }
        }
        return (min_cell, max_cell);
    }
    (
        segment
            .min_row
            .and_then(|row| views.get(row).copied().flatten()),
        segment
            .max_row
            .and_then(|row| views.get(row).copied().flatten()),
    )
}

/// Checks the batch covers the schema exactly and enforces NOT NULL at
/// the format boundary. Returns the row count
fn validate_batch(schema: &LakeSchema, columns: &[ColumnData]) -> Result<usize, ZyronError> {
    if columns.len() != schema.columns.len() {
        return Err(ZyronError::Internal(format!(
            "batch has {} columns, schema has {}",
            columns.len(),
            schema.columns.len()
        )));
    }
    let mut row_count = None;
    for data in columns {
        let col = schema.column_by_id(data.column_id).ok_or_else(|| {
            ZyronError::Internal(format!(
                "batch column {} is not in the schema",
                data.column_id
            ))
        })?;
        match row_count {
            None => row_count = Some(data.cells.len()),
            Some(n) if n != data.cells.len() => {
                return Err(ZyronError::Internal(format!(
                    "column \"{}\" has {} rows, expected {}",
                    col.name,
                    data.cells.len(),
                    n
                )));
            }
            Some(_) => {}
        }
        if !col.nullable {
            if let Some(row) = data.cells.iter().position(|c| c.is_none()) {
                return Err(ZyronError::Internal(format!(
                    "null value in non-nullable column \"{}\" at row {}",
                    col.name, row
                )));
            }
        }
    }
    match row_count {
        Some(0) | None => Err(ZyronError::Internal("empty row batch".into())),
        Some(n) => Ok(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::cell_to_value;
    use crate::manifest::FileStats;
    use crate::predicate::{CompareOp, LakePredicate, LakeValue, PruneDecision, StatsSource};
    use crate::schema::LakeColumn;
    use zyron_common::TypeId;

    fn column(id: u32, name: &str, type_id: TypeId) -> LakeColumn {
        LakeColumn {
            id,
            name: name.into(),
            type_id,
            nullable: true,
            fractional_digits: None,
            tz_offset_secs: None,
            max_length: None,
            default_expr: None,
        }
    }

    fn cells<const N: usize>(values: [&[u8]; N]) -> Vec<Option<Vec<u8>>> {
        values.iter().map(|v| Some(v.to_vec())).collect()
    }

    #[test]
    fn test_a_declared_curve_changes_the_file_layout() {
        use crate::curve::{normalize_component, ordering_key};
        use crate::manifest::ClusterStrategy;
        use crate::reader::LakeFileReader;

        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 61);
        let schema = LakeSchema::new(
            1,
            vec![column(0, "x", TypeId::Int64), column(1, "y", TypeId::Int64)],
        )
        .expect("schema");

        // A grid, so the two curves have something to disagree about
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for x in 0..8i64 {
            for y in 0..8i64 {
                xs.push(Some(x.to_le_bytes().to_vec()));
                ys.push(Some(y.to_le_bytes().to_vec()));
            }
        }
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: xs,
            },
            ColumnData {
                column_id: 1,
                cells: ys,
            },
        ];

        let write = |partition_id: u64, strategy: ClusterStrategy| {
            write_data_file(
                &paths,
                &schema,
                &WriteRequest {
                    partition_id,
                    columns: &columns,
                    sort_keys: &[0, 1],
                    sort_strategies: &[strategy, strategy],
                    cluster_spec_id: 1,
                    table_id: 61,
                    bloom_columns: &[],
                    index_id: None,
                },
            )
            .expect("write")
        };
        write(0x71, ClusterStrategy::RangePartition);
        write(0x72, ClusterStrategy::BitInterleave);

        // Read each file's rows back in stored order
        let read = |partition_id: u64| -> Vec<(i64, i64)> {
            let reader = LakeFileReader::open(&paths, partition_id).expect("open");
            let x = reader
                .read_column(schema.column_by_id(0).expect("x"))
                .expect("x");
            let y = reader
                .read_column(schema.column_by_id(1).expect("y"))
                .expect("y");
            (0..reader.row_count())
                .map(|row| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(x.cell(row).expect("cell"));
                    let mut b = [0u8; 8];
                    b.copy_from_slice(y.cell(row).expect("cell"));
                    (i64::from_le_bytes(a), i64::from_le_bytes(b))
                })
                .collect()
        };
        let ranged = read(0x71);
        let interleaved = read(0x72);
        assert_eq!(ranged.len(), 64);
        assert_eq!(interleaved.len(), 64);
        assert_ne!(
            ranged, interleaved,
            "a declared curve that did not change the layout would be a note and nothing else"
        );

        // Range order is plain ascending by the leading column
        assert_eq!(ranged[0], (0, 0));
        assert_eq!(ranged[7], (0, 7));
        assert_eq!(ranged[8], (1, 0));

        // The interleaved file matches what the curve says, row for row
        let mut expected: Vec<(i64, i64)> = ranged.clone();
        expected.sort_by_key(|(x, y)| {
            ordering_key(
                ClusterStrategy::BitInterleave,
                &[
                    normalize_component(TypeId::Int64, &x.to_le_bytes()),
                    normalize_component(TypeId::Int64, &y.to_le_bytes()),
                ],
            )
        });
        assert_eq!(interleaved, expected);
    }

    /// The manifest entry's bounds, null count and distinct estimate all
    /// come off the segment build now, so they have to be what a flat scan
    /// of the same cells says. A bound that drifted would prune away files
    /// that hold matching rows.
    ///
    /// Every ordered family is covered, including the fixed-width byte
    /// family whose stat slots order by the little endian reading and whose
    /// cells order by their bytes, which is the one shape the writer still
    /// compares itself
    #[test]
    fn test_manifest_bounds_match_a_flat_scan_of_the_cells() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 91);
        let schema = LakeSchema::new(
            1,
            vec![
                column(0, "signed", TypeId::Int64),
                column(1, "unsigned", TypeId::UInt64),
                column(2, "real", TypeId::Float64),
                column(3, "text", TypeId::Varchar),
                column(4, "id", TypeId::Uuid),
                column(5, "flag", TypeId::Boolean),
                column(6, "ts", TypeId::Timestamp),
                column(7, "sparse", TypeId::Int32),
                // An unordered family records no bounds at all
                column(8, "doc", TypeId::Json),
            ],
        )
        .expect("schema");

        // Enough rows to span several zones, scattered so the extremes do
        // not land at either end of the file
        let rows = 5_000usize;
        let scatter = |i: usize| (i * 2_731) % rows;
        let cells_for = |id: u32| -> Vec<Option<Vec<u8>>> {
            (0..rows)
                .map(|i| {
                    let n = scatter(i);
                    match id {
                        0 => Some(((n as i64) - 2_500).to_le_bytes().to_vec()),
                        1 => Some((u64::MAX - n as u64).to_le_bytes().to_vec()),
                        2 => Some(((n as f64) - 2_500.5).to_le_bytes().to_vec()),
                        3 => {
                            let mut value = vec![b'p'; 32];
                            value.extend_from_slice(format!("{:08}", n).as_bytes());
                            Some(value)
                        }
                        4 => {
                            let mut id = [0u8; 16];
                            id[..8].copy_from_slice(&(n as u64).to_be_bytes());
                            id[8..].copy_from_slice(&(n as u64).to_le_bytes());
                            Some(id.to_vec())
                        }
                        5 => Some(vec![(n % 2) as u8]),
                        6 => Some(
                            (((n as i64) - 2_500) * 86_400_000_000)
                                .to_le_bytes()
                                .to_vec(),
                        ),
                        7 => {
                            if (1_024..2_048).contains(&i) || i % 7 == 0 {
                                None
                            } else {
                                Some(((n as i32) - 2_500).to_le_bytes().to_vec())
                            }
                        }
                        _ => Some(format!("{{\"n\":{}}}", n).into_bytes()),
                    }
                })
                .collect()
        };
        let columns: Vec<ColumnData> = schema
            .columns
            .iter()
            .map(|col| ColumnData {
                column_id: col.id,
                cells: cells_for(col.id),
            })
            .collect();

        // Written under a sort key, so the stored order is a permutation of
        // the input and the rows the segment reports are file rows
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x91,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[ClusterStrategy::RangePartition],
                cluster_spec_id: 0,
                table_id: 91,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write");

        for data in &columns {
            let col = schema.column_by_id(data.column_id).expect("column");
            let physical = col.physical_type_id();
            let name = &col.name;
            let stats = entry.stats_for(data.column_id).expect("stats");

            let mut min_cell: Option<&[u8]> = None;
            let mut max_cell: Option<&[u8]> = None;
            let mut nulls = 0u64;
            let mut distinct = std::collections::HashSet::new();
            for cell in &data.cells {
                match cell {
                    Some(cell) => {
                        distinct.insert(cell.as_slice());
                        if min_cell.is_none_or(|m| compare_cells(physical, cell, m).is_lt()) {
                            min_cell = Some(cell);
                        }
                        if max_cell.is_none_or(|m| compare_cells(physical, cell, m).is_gt()) {
                            max_cell = Some(cell);
                        }
                    }
                    None => nulls += 1,
                }
            }

            assert_eq!(
                stats.bounds.min,
                min_cell.and_then(|c| cell_to_value(physical, c)),
                "column \"{name}\" recorded a minimum the cells do not agree with"
            );
            assert_eq!(
                stats.bounds.max,
                max_cell.and_then(|c| cell_to_value(physical, c)),
                "column \"{name}\" recorded a maximum the cells do not agree with"
            );
            assert_eq!(
                stats.bounds.null_count, nulls,
                "column \"{name}\" recorded the wrong null count"
            );
            assert_eq!(stats.bounds.row_count, rows as u64);

            let ndv = stats
                .ndv
                .unwrap_or_else(|| panic!("column \"{name}\" carries no distinct estimate"));
            let truth = distinct.len() as f64;
            let error = (ndv as f64 - truth).abs() / truth;
            assert!(
                error < 0.05,
                "column \"{name}\" estimated {ndv} distinct against {truth}"
            );
        }

        // An unordered family has no comparison, so it records no bounds
        let doc = entry.stats_for(8).expect("stats");
        assert!(doc.bounds.min.is_none() && doc.bounds.max.is_none());
    }

    /// A manifest entry has to stay metadata.
    ///
    /// A bloom is ten bits per value, so copying one into the entry for
    /// every column that clears the cardinality heuristic makes the manifest
    /// grow with the rows in the table rather than the files in it, and the
    /// manifest is held in memory per version. An undeclared column keeps
    /// its filter in the segment, where a probe that has opened the file
    /// still finds it, and the entry carries nothing for it.
    #[test]
    fn test_only_a_declared_bloom_column_puts_its_filter_in_the_manifest() {
        use zyron_storage::columnar::ZyrFileReader;

        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 77);
        let schema = LakeSchema::new(
            1,
            vec![
                column(0, "declared", TypeId::Int64),
                column(1, "undeclared", TypeId::Int64),
            ],
        )
        .expect("schema");

        // Both columns are high cardinality, so the heuristic would build a
        // filter for each of them
        let rows = 4096usize;
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: (0..rows)
                    .map(|i| Some((i as i64).to_le_bytes().to_vec()))
                    .collect(),
            },
            ColumnData {
                column_id: 1,
                cells: (0..rows)
                    .map(|i| Some(((i as i64) * 31).to_le_bytes().to_vec()))
                    .collect(),
            },
        ];
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x77,
                columns: &columns,
                sort_keys: &[],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 77,
                bloom_columns: &[0],
                index_id: None,
            },
        )
        .expect("write");

        let declared = entry.stats_for(0).expect("declared stats");
        let undeclared = entry.stats_for(1).expect("undeclared stats");
        assert!(
            declared.bloom.is_some(),
            "a declared column buys the probe that answers before the file is opened"
        );
        assert!(
            undeclared.bloom.is_none(),
            "an undeclared column must not spend manifest bytes on a filter"
        );

        // The undeclared column still has its filter in the file, so a probe
        // that has opened it loses nothing
        let reader = ZyrFileReader::open(&paths.data_file(0x77)).expect("open");
        for column_id in [0u32, 1] {
            let bloom = reader
                .read_bloom(column_id)
                .expect("read bloom")
                .unwrap_or_else(|| panic!("column {column_id} carries no filter in its segment"));
            let data = columns
                .iter()
                .find(|c| c.column_id == column_id)
                .expect("column");
            for cell in data.cells.iter().flatten() {
                assert!(
                    bloom.might_contain(cell.as_slice()),
                    "column {column_id} filter denies a value the segment stored"
                );
            }
        }

        // And the manifest still prunes on the declared column, which is
        // what the bytes were spent for
        let stats = FileStats::new(&entry, &schema);
        assert!(stats.may_contain(0, &LakeValue::Int(40)));
        assert!(
            !stats.may_contain(0, &LakeValue::Int(rows as i64 + 5)),
            "a value the declared column never stored is provably absent"
        );
        // The undeclared column prunes nothing from the manifest, which is a
        // missed skip and never a dropped row
        assert!(stats.may_contain(1, &LakeValue::Int(7)));
    }

    /// A bloom answering "absent" for a value the writer stored would drop
    /// rows that exist, so the bytes the prune path probes with must be the
    /// bytes ColumnSegment::build inserted, for every type family.
    #[test]
    fn test_lake_bloom_canonical_bytes_match_segment_build() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 1);
        let schema = LakeSchema::new(
            1,
            vec![
                column(0, "i64", TypeId::Int64),
                column(1, "i32", TypeId::Int32),
                column(2, "u64", TypeId::UInt64),
                column(3, "f64", TypeId::Float64),
                column(4, "f32", TypeId::Float32),
                column(5, "text", TypeId::Varchar),
                column(6, "id", TypeId::Uuid),
                column(7, "flag", TypeId::Boolean),
                column(8, "ts", TypeId::Timestamp),
            ],
        )
        .expect("schema");

        let uuids: Vec<Option<Vec<u8>>> = (0u8..4).map(|n| Some([n; 16].to_vec())).collect();
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: cells([
                    &(-9_000_000_000i64).to_le_bytes(),
                    &0i64.to_le_bytes(),
                    &42i64.to_le_bytes(),
                    &i64::MAX.to_le_bytes(),
                ]),
            },
            ColumnData {
                column_id: 1,
                cells: cells([
                    &(-7i32).to_le_bytes(),
                    &1i32.to_le_bytes(),
                    &900i32.to_le_bytes(),
                    &i32::MIN.to_le_bytes(),
                ]),
            },
            ColumnData {
                column_id: 2,
                cells: cells([
                    &0u64.to_le_bytes(),
                    &7u64.to_le_bytes(),
                    &u64::MAX.to_le_bytes(),
                    &1234u64.to_le_bytes(),
                ]),
            },
            ColumnData {
                column_id: 3,
                cells: cells([
                    &(-1.5f64).to_le_bytes(),
                    &0.0f64.to_le_bytes(),
                    &3.25f64.to_le_bytes(),
                    &1e300f64.to_le_bytes(),
                ]),
            },
            ColumnData {
                column_id: 4,
                cells: cells([
                    &(-1.5f32).to_le_bytes(),
                    &0.5f32.to_le_bytes(),
                    &7.75f32.to_le_bytes(),
                    &100.0f32.to_le_bytes(),
                ]),
            },
            ColumnData {
                column_id: 5,
                cells: cells([b"alice", b"bob\0\0", b"", b"carol"]),
            },
            ColumnData {
                column_id: 6,
                cells: uuids,
            },
            ColumnData {
                column_id: 7,
                cells: cells([&[0u8], &[1u8], &[1u8], &[0u8]]),
            },
            ColumnData {
                column_id: 8,
                cells: cells([
                    &(-86_400_000_000i64).to_le_bytes(),
                    &0i64.to_le_bytes(),
                    &1_700_000_000_000_000i64.to_le_bytes(),
                    &99i64.to_le_bytes(),
                ]),
            },
        ];

        // Force a filter on every column so the probe is exercised even where
        // the cardinality heuristic would skip one
        let bloom_columns: Vec<u32> = schema.columns.iter().map(|c| c.id).collect();
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x51,
                columns: &columns,
                sort_keys: &[],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 1,
                bloom_columns: &bloom_columns,
                index_id: None,
            },
        )
        .expect("write");

        let stats = FileStats::new(&entry, &schema);
        for data in &columns {
            let col = schema.column_by_id(data.column_id).expect("column");
            let file_stats = entry.stats_for(data.column_id).expect("stats");
            assert!(
                file_stats.bloom.is_some(),
                "column {} must carry the forced bloom",
                col.name
            );
            for cell in data.cells.iter().flatten() {
                let value = cell_to_value(col.physical_type_id(), cell)
                    .unwrap_or_else(|| panic!("column {} cell has no value", col.name));
                assert!(
                    stats.may_contain(data.column_id, &value),
                    "column {} bloom denies a value it stored: {:?}",
                    col.name,
                    value
                );
            }
        }
    }

    /// The bloom is actually consulted: a constant inside the column min/max
    /// that the file never stored prunes the file with no IO.
    #[test]
    fn test_bloom_prunes_a_value_inside_the_bounds() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 2);
        let schema = LakeSchema::new(1, vec![column(0, "k", TypeId::Int64)]).expect("schema");
        let columns = vec![ColumnData {
            column_id: 0,
            cells: (0..512i64)
                .map(|i| Some((i * 10).to_le_bytes().to_vec()))
                .collect(),
        }];
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x52,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 2,
                bloom_columns: &[0],
                index_id: None,
            },
        )
        .expect("write");

        let stats = FileStats::new(&entry, &schema);
        let present = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Eq,
            value: LakeValue::Int(1230),
        };
        assert_eq!(present.prune(&stats), PruneDecision::MayMatch);

        // Every multiple of ten is stored, so a gap value sits inside the
        // bounds and is provably absent
        let absent = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Eq,
            value: LakeValue::Int(1231),
        };
        assert_eq!(absent.prune(&stats), PruneDecision::CannotMatch);

        // Bounds alone cannot reach that answer
        assert_eq!(absent.prune(&entry), PruneDecision::MayMatch);
    }

    /// A constant with no provable stored form must prune nothing rather
    /// than probe with bytes that are not the column cells.
    #[test]
    fn test_unrepresentable_constant_prunes_nothing() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 3);
        let schema = LakeSchema::new(1, vec![column(0, "k", TypeId::Int32)]).expect("schema");
        let columns = vec![ColumnData {
            column_id: 0,
            cells: (0..300i32)
                .map(|v| Some(v.to_le_bytes().to_vec()))
                .collect(),
        }];
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x53,
                columns: &columns,
                sort_keys: &[],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 3,
                bloom_columns: &[0],
                index_id: None,
            },
        )
        .expect("write");

        let stats = FileStats::new(&entry, &schema);
        // A string constant against an integer column has no cell form
        assert!(stats.may_contain(0, &LakeValue::Str("100".into())));
    }
}

//! Data file writer.
//!
//! Turns one sorted row batch into one immutable .zyr file at the lake's
//! partition-addressed path and returns the manifest entry describing it.
//! Lake files carry only user columns, visibility is by log version, so
//! there are no MVCC system columns and no patch store.
//!
//! Statistics are exact. The writer materializes every column anyway, so
//! min, max and null count come from a full pass, not a sample, and the
//! segment's value bloom is copied into the manifest entry so pruning
//! probes it with zero IO

use std::path::Path;

use zyron_common::ZyronError;
use zyron_storage::columnar::{
    BloomPolicy, ColumnSegment, SegmentOptions, SortOrder, ZyrFileHeader, ZyrFileWriter,
};

use crate::cells::{CellFamily, cell_family, cell_to_value, compare_cells};
use crate::curve::{normalize_component, ordering_key};
use crate::hll::DistinctSketch;
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

    // Sort routing: a permutation over row indices ordered by the key
    // columns under the curve each one declared, nulls last, stable for
    // determinism. A declared strategy that did not change the order would
    // be a note in the manifest and nothing else.
    //
    // The curve is the first key's, because a multi-dimensional curve
    // consumes every dimension at once and keys declared after it only
    // refine rows the curve placed together. It decides both the
    // permutation below and whether the header may claim ascending order
    let leading_strategy = req
        .sort_strategies
        .first()
        .copied()
        .unwrap_or(ClusterStrategy::RangePartition);
    let order: Vec<usize> = if req.sort_keys.is_empty() {
        (0..row_count).collect()
    } else {
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
        // One ordering key per row, built once rather than per comparison:
        // a sort does O(n log n) comparisons and this makes each of them a
        // byte compare over `8 * dimensions` bytes
        let mut ordering: Vec<(bool, Vec<u8>)> = Vec::with_capacity(row_count);
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
            ordering.push((any_null, ordering_key(leading_strategy, &axes)));
        }

        let mut order: Vec<usize> = (0..row_count).collect();
        order.sort_by(|&a, &b| {
            ordering[a]
                .0
                .cmp(&ordering[b].0)
                .then_with(|| ordering[a].1.cmp(&ordering[b].1))
        });
        order
    };

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

        let mut views: Vec<Option<&[u8]>> = Vec::with_capacity(row_count);
        let mut null_count = 0u64;
        let mut min_cell: Option<&[u8]> = None;
        let mut max_cell: Option<&[u8]> = None;
        // Distinct values, counted in the pass that already materializes
        // the column. The sketch is 1 KiB and is dropped below, only its
        // estimate reaches the manifest
        let mut distinct = DistinctSketch::new();
        let orderable = cell_family(physical) != CellFamily::Unordered;
        for &row in &order {
            match &data.cells[row] {
                Some(cell) => {
                    let cell = cell.as_slice();
                    distinct.insert(cell);
                    if orderable {
                        if min_cell
                            .map(|m| compare_cells(physical, cell, m).is_lt())
                            .unwrap_or(true)
                        {
                            min_cell = Some(cell);
                        }
                        if max_cell
                            .map(|m| compare_cells(physical, cell, m).is_gt())
                            .unwrap_or(true)
                        {
                            max_cell = Some(cell);
                        }
                    }
                    views.push(Some(cell));
                }
                None => {
                    null_count += 1;
                    views.push(None);
                }
            }
        }

        // The whole column is materialized here already, so the encoding
        // is picked by trial encoding every row rather than a prefix
        let options = SegmentOptions {
            bloom: if req.bloom_columns.contains(&col.id) {
                BloomPolicy::Force
            } else {
                BloomPolicy::Auto
            },
            exact_encoding: true,
        };
        let segment =
            ColumnSegment::build_with_options(col.id, physical, value_size, &views, options)?;
        let zone_bytes: Vec<u8> = segment
            .zone_maps
            .iter()
            .flat_map(|z| z.to_bytes())
            .collect();
        let bloom_bytes = segment.bloom_filter.as_ref().map(|b| b.to_bytes());
        writer.write_segment(
            col.id,
            &segment.header.to_bytes(),
            bloom_bytes.as_deref(),
            &zone_bytes,
            &segment.null_bitmap,
            &segment.encoded_data,
        )?;

        column_stats.push(ColumnStatsEntry {
            column_id: col.id,
            bounds: ColumnBounds {
                min: min_cell.and_then(|c| cell_to_value(physical, c)),
                max: max_cell.and_then(|c| cell_to_value(physical, c)),
                null_count,
                row_count: row_count as u64,
            },
            bloom: bloom_bytes,
            ndv: Some(distinct.estimate()),
        });
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
            column_stats,
            delete_predicate_ids: Vec::new(),
        },
        order,
    })
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

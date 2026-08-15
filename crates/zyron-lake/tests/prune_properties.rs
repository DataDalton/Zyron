//! Properties the pruning path has to hold whatever the build profile.
//!
//! These are not timing measurements. Every assertion here is a counted
//! fact: allocations that did or did not happen, files that were or were
//! not opened, a survivor mask compared byte for byte against the exact
//! reference. That makes them meaningful in an unoptimized build, which
//! is where they run, and it makes a regression a failure rather than a
//! number somebody has to interpret.
//!
//! Allocation is counted per thread, so these tests measure only their
//! own work however many other tests run beside them.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs;

use zyron_common::TypeId;
use zyron_lake::manifest::{ClusterSpec, ColumnStatsEntry, PartitionEntry};
use zyron_lake::predicate::ColumnBounds;
use zyron_lake::{
    ColumnData, CommitAttempt, CompareOp, GateConfig, LakeColumn, LakePaths, LakePredicate,
    LakeSchema, LakeValue, LogEntry, ManifestFile, OperationKind, PredicateClass, PruneDecision,
    PruneIndex, StoredFilter, TransactionLog, WriteRequest, current_epoch, evaluate, observe_scan,
    observe_scan_result, skip_rate, with_sweep,
};

thread_local! {
    static LOCAL_ALLOCS: Cell<usize> = const { Cell::new(0) };
    static LOCAL_BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Counts allocations on the thread that made them.
///
/// A process-wide counter would be polluted by every other test running
/// beside these, so the counter is thread local. `try_with` is used
/// because the allocator is also called while thread locals are being
/// destroyed, when the counter is no longer reachable
struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size.saturating_sub(layout.size()));
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn record(size: usize) {
    let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
    let _ = LOCAL_BYTES.try_with(|c| c.set(c.get() + size));
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` and reports what it allocated on this thread
fn measure<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    let allocs = LOCAL_ALLOCS.with(|c| c.get());
    let bytes = LOCAL_BYTES.with(|c| c.get());
    let out = f();
    (
        out,
        LOCAL_ALLOCS.with(|c| c.get()) - allocs,
        LOCAL_BYTES.with(|c| c.get()) - bytes,
    )
}

fn schema(types: &[(&str, TypeId)]) -> LakeSchema {
    LakeSchema::new(
        1,
        types
            .iter()
            .enumerate()
            .map(|(i, (name, type_id))| LakeColumn {
                id: i as u32,
                name: (*name).into(),
                type_id: *type_id,
                nullable: true,
                fractional_digits: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            })
            .collect(),
    )
    .expect("valid schema")
}

fn int_stats(column_id: u32, min: i64, max: i64, rows: u64) -> ColumnStatsEntry {
    ColumnStatsEntry {
        column_id,
        bounds: ColumnBounds {
            min: Some(LakeValue::Int(min)),
            max: Some(LakeValue::Int(max)),
            null_count: 0,
            row_count: rows,
        },
        bloom: None,
        ndv: Some(rows),
        size_bytes: None,
    }
}

/// A manifest of `files` data files, each holding a disjoint hundred-wide
/// run of `id` and a `bucket` column that repeats every sixteen files, so
/// a predicate on either one prunes a different shape of the file set
fn synthetic_manifest(files: usize) -> ManifestFile {
    let entries = (0..files)
        .map(|i| {
            let lo = i as i64 * 100;
            let bucket = (i % 16) as i64;
            PartitionEntry {
                partition_id: i as u64,
                size_bytes: 1 << 20,
                row_count: 100,
                added_version: 1,
                cluster_spec_id: 0,
                column_stats: vec![
                    int_stats(0, lo, lo + 99, 100),
                    int_stats(1, bucket, bucket, 100),
                ],
                delete_predicate_ids: Vec::new(),
            }
        })
        .collect();
    ManifestFile {
        snapshot_id: 1,
        parent_snapshot_id: 0,
        timestamp_us: 0,
        schema: schema(&[("id", TypeId::Int64), ("bucket", TypeId::Int64)]),
        cluster_spec: ClusterSpec::none(),
        entries,
        delete_predicates: Vec::new(),
        properties: BTreeMap::new(),
        indexes: Vec::new(),
        index_files: Vec::new(),
    }
}

fn cmp(column_id: u32, op: CompareOp, value: i64) -> LakePredicate {
    LakePredicate::Compare {
        column_id,
        op,
        value: LakeValue::Int(value),
    }
}

/// The exact per-file answer, which every sweep is measured against
fn scalar_mask(manifest: &ManifestFile, predicate: &LakePredicate) -> Vec<u8> {
    manifest
        .entries
        .iter()
        .map(|e| (manifest.prune_file(predicate, e) == PruneDecision::CannotMatch) as u8)
        .collect()
}

/// Predicate shapes covering every node kind, including the negations
/// that used to be pruned by rewriting the tree
fn predicate_catalog() -> Vec<LakePredicate> {
    vec![
        cmp(0, CompareOp::Eq, 4_050),
        cmp(0, CompareOp::Lt, 100_000),
        cmp(0, CompareOp::GtEq, 9_000_000),
        cmp(1, CompareOp::Eq, 7),
        LakePredicate::And(vec![
            cmp(0, CompareOp::GtEq, 1_000_000),
            cmp(1, CompareOp::Eq, 3),
        ]),
        LakePredicate::Or(vec![
            cmp(0, CompareOp::Lt, 5_000),
            cmp(0, CompareOp::GtEq, 9_990_000),
        ]),
        LakePredicate::Not(Box::new(LakePredicate::And(vec![
            cmp(0, CompareOp::GtEq, 0),
            cmp(0, CompareOp::Lt, 5_000_000),
        ]))),
        LakePredicate::In {
            column_id: 1,
            values: vec![LakeValue::Int(2), LakeValue::Int(11)],
        },
        LakePredicate::Not(Box::new(LakePredicate::In {
            column_id: 1,
            values: vec![LakeValue::Int(2)],
        })),
        LakePredicate::IsNotNull { column_id: 0 },
    ]
}

/// Deciding which files a predicate cannot match reads statistics that are
/// already in memory, so it must not allocate at all. A Not used to be
/// pruned by cloning and rewriting its subtree, which allocated once per
/// file and is exactly what this catches
#[test]
fn test_zero_allocation_in_predicate_eval_loop() {
    let manifest = synthetic_manifest(4_096);
    let predicates = predicate_catalog();

    // One untimed pass so nothing lazily initialized is charged to the
    // measured one
    for p in &predicates {
        let _ = manifest.files_matching(p).count();
    }

    for predicate in &predicates {
        let (survivors, allocs, bytes) = measure(|| manifest.files_matching(predicate).count());
        assert!(survivors <= manifest.entries.len());
        assert_eq!(
            (allocs, bytes),
            (0, 0),
            "pruning {:?} across {} files allocated {} times for {} bytes",
            predicate,
            manifest.entries.len(),
            allocs,
            bytes
        );
    }
}

/// The vectorized sweep holds its working bytes per thread, so a warm
/// thread prunes without allocating however wide the predicate
#[test]
fn test_zero_allocation_in_prune_index_sweep() {
    let manifest = synthetic_manifest(4_096);
    let index = PruneIndex::build(&manifest);
    let predicates = predicate_catalog();

    for p in &predicates {
        with_sweep(&index, p, |_, _| {});
    }

    for predicate in &predicates {
        let (_, allocs, bytes) = measure(|| {
            with_sweep(&index, predicate, |mask, _| {
                mask.iter().filter(|b| **b == 0).count()
            })
        });
        assert_eq!(
            (allocs, bytes),
            (0, 0),
            "sweeping {:?} allocated {} times for {} bytes",
            predicate,
            allocs,
            bytes
        );
    }
}

/// The sweep is a proof, never a guess. Every file it marks is one the
/// exact statistics also reject, and on losslessly keyed columns with no
/// value bloom the two masks are identical byte for byte
#[test]
fn test_prune_index_survivor_mask_matches_scalar_reference() {
    const FILES: usize = 100_000;
    let manifest = synthetic_manifest(FILES);
    let index = PruneIndex::build(&manifest);
    assert_eq!(index.file_count(), FILES);
    assert_eq!(index.indexed_columns(), &[0, 1]);

    for predicate in predicate_catalog() {
        let expected = scalar_mask(&manifest, &predicate);
        with_sweep(&index, &predicate, |mask, complete| {
            assert!(
                complete,
                "int columns with no bloom decide exactly: {:?}",
                predicate
            );
            assert_eq!(
                mask,
                &expected[..],
                "sweep and exact reference disagree for {:?}",
                predicate
            );
        });
    }

    // At least one predicate has to actually prune, or the comparison
    // above would pass on a sweep that never rejects anything
    let selective = cmp(0, CompareOp::Lt, 1_000);
    let kept = manifest.files_matching(&selective).count();
    assert_eq!(
        kept, 10,
        "a narrow range must reach ten of a hundred thousand files"
    );
}

/// Deciding which files to skip reads the manifest and nothing else, so
/// it holds with every data file removed from disk. A pruning path that
/// opened a file to answer would fail here rather than merely be slow
#[test]
fn test_pruning_touches_no_data_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 11);
    let schema = schema(&[("id", TypeId::Int64)]);
    let log = new_log(&paths, &schema);

    let mut written = Vec::new();
    for file in 0..4u64 {
        let cells: Vec<Option<Vec<u8>>> = (0..64i64)
            .map(|r| Some((file as i64 * 1_000 + r).to_le_bytes().to_vec()))
            .collect();
        let columns = vec![ColumnData {
            column_id: 0,
            cells,
        }];
        let entry = zyron_lake::write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: file,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 11,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write data file");
        written.push(entry);
    }
    log.commit(attempt(OperationKind::Append), |_| {
        Ok(written.iter().cloned().map(LogEntry::AddFile).collect())
    })
    .expect("append");

    let manifest = log.latest_manifest().expect("manifest");
    let index = log.latest_prune_index().expect("prune index");
    let predicate = cmp(0, CompareOp::Lt, 1_000);
    let expected: Vec<u64> = manifest
        .files_matching(&predicate)
        .map(|e| e.partition_id)
        .collect();
    assert_eq!(
        expected,
        vec![0],
        "only the first file holds ids below 1000"
    );

    // Every data file goes, so any read of one is now an error rather
    // than a slower path that still returns the right answer
    let mut removed = 0;
    for dirent in fs::read_dir(paths.data_dir()).expect("data dir") {
        let path = dirent.expect("dirent").path();
        fs::remove_file(&path).expect("remove data file");
        removed += 1;
    }
    assert_eq!(removed, 4, "every data file was removed");

    let after: Vec<u64> = manifest
        .files_matching(&predicate)
        .map(|e| e.partition_id)
        .collect();
    assert_eq!(after, expected, "exact pruning read no data file");

    with_sweep(&index, &predicate, |mask, complete| {
        assert!(complete);
        assert_eq!(
            mask,
            &[0, 1, 1, 1][..],
            "the sweep read no data file either"
        );
    });
}

/// Statistics are complete the moment a file is written, so the manifest
/// is the statistics source rather than a cache of one and no ANALYZE
/// ever runs against a lake table. `lake_ddl_test` asserts the other end
/// of this, that a plan reads those numbers
#[test]
fn test_written_file_carries_its_own_statistics_with_no_analyze() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 12);
    let schema = schema(&[("id", TypeId::Int64), ("label", TypeId::Varchar)]);

    let ids: Vec<Option<Vec<u8>>> = (0..500i64)
        .map(|r| Some((r % 50).to_le_bytes().to_vec()))
        .collect();
    let labels: Vec<Option<Vec<u8>>> = (0..500i64)
        .map(|r| {
            if r % 10 == 0 {
                None
            } else {
                Some(format!("label-{:04}", r % 25).into_bytes())
            }
        })
        .collect();
    let columns = vec![
        ColumnData {
            column_id: 0,
            cells: ids,
        },
        ColumnData {
            column_id: 1,
            cells: labels,
        },
    ];
    let entry = zyron_lake::write_data_file(
        &paths,
        &schema,
        &WriteRequest {
            partition_id: 1,
            columns: &columns,
            sort_keys: &[0],
            sort_strategies: &[],
            cluster_spec_id: 0,
            table_id: 12,
            bloom_columns: &[],
            index_id: None,
        },
    )
    .expect("write data file");

    // Bounds and null counts are counted, so they are exact. The distinct
    // count comes from a sketch and is documented as within a few percent,
    // which is the accuracy an estimator needs and far more than a stale
    // ANALYZE would give
    assert_eq!(entry.row_count, 500);
    let id_stats = entry.stats_for(0).expect("id statistics");
    assert_eq!(id_stats.bounds.min, Some(LakeValue::Int(0)));
    assert_eq!(id_stats.bounds.max, Some(LakeValue::Int(49)));
    assert_eq!(id_stats.bounds.null_count, 0);
    assert_within_a_few_percent(id_stats.ndv, 50, "id");

    let label_stats = entry.stats_for(1).expect("label statistics");
    assert_eq!(
        label_stats.bounds.min,
        Some(LakeValue::Str("label-0000".into()))
    );
    assert_eq!(
        label_stats.bounds.max,
        Some(LakeValue::Str("label-0024".into()))
    );
    assert_eq!(label_stats.bounds.null_count, 50);
    assert_within_a_few_percent(label_stats.ndv, 25, "label");
}

fn assert_within_a_few_percent(estimate: Option<u64>, truth: u64, column: &str) {
    let estimate =
        estimate.unwrap_or_else(|| panic!("column {} carries no distinct count", column));
    let error = (estimate as f64 - truth as f64).abs() / truth as f64;
    assert!(
        error <= 0.05,
        "column {} estimated {} distinct values against {}, {:.1}% off",
        column,
        estimate,
        truth,
        error * 100.0
    );
}

/// The published version is an atomic load. Every filesystem read in this
/// crate builds a path or a buffer first, so a read that allocated
/// nothing opened nothing
#[test]
fn test_get_latest_version_performs_no_io() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 13);
    let schema = schema(&[("id", TypeId::Int64)]);
    let log = new_log(&paths, &schema);
    log.commit(attempt(OperationKind::Append), |_| {
        Ok(vec![LogEntry::AddFile(PartitionEntry {
            partition_id: 1,
            size_bytes: 512,
            row_count: 64,
            added_version: 0,
            cluster_spec_id: 0,
            column_stats: vec![int_stats(0, 0, 63, 64)],
            delete_predicate_ids: Vec::new(),
        })])
    })
    .expect("append");

    let _ = log.latest_version();
    let (versions, allocs, bytes) = measure(|| {
        let mut seen = 0u64;
        for _ in 0..100_000 {
            seen = seen.max(log.latest_version());
        }
        seen
    });
    assert_eq!(versions, 2);
    assert_eq!(
        (allocs, bytes),
        (0, 0),
        "reading the published version allocated {} times for {} bytes",
        allocs,
        bytes
    );
}

/// A narrow projection reads its own column and no other. The wide column
/// is written so it cannot compress away, so the difference between
/// reading one and reading both is the wide column's payload
#[test]
fn test_point_lookup_reads_only_projected_columns() {
    const ROWS: usize = 512;
    const BLOB_LEN: usize = 256;

    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 14);
    let schema = schema(&[("id", TypeId::Int64), ("blob", TypeId::Varchar)]);

    // A counter-driven byte pattern, distinct per row and per position, so
    // dictionary and run encodings have nothing to collapse
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let ids: Vec<Option<Vec<u8>>> = (0..ROWS as i64)
        .map(|r| Some(r.to_le_bytes().to_vec()))
        .collect();
    let blobs: Vec<Option<Vec<u8>>> = (0..ROWS)
        .map(|_| {
            let mut cell = Vec::with_capacity(BLOB_LEN);
            while cell.len() < BLOB_LEN {
                cell.extend_from_slice(&next().to_le_bytes().map(|b| b'!' + (b % 90)));
            }
            cell.truncate(BLOB_LEN);
            Some(cell)
        })
        .collect();
    let columns = vec![
        ColumnData {
            column_id: 0,
            cells: ids,
        },
        ColumnData {
            column_id: 1,
            cells: blobs,
        },
    ];
    zyron_lake::write_data_file(
        &paths,
        &schema,
        &WriteRequest {
            partition_id: 5,
            columns: &columns,
            sort_keys: &[0],
            sort_strategies: &[],
            cluster_spec_id: 0,
            table_id: 14,
            bloom_columns: &[],
            index_id: None,
        },
    )
    .expect("write data file");

    let reader = zyron_lake::LakeFileReader::open(&paths, 5).expect("open data file");
    let id_column = schema.column_by_id(0).expect("id column");
    let blob_column = schema.column_by_id(1).expect("blob column");

    // One untimed read of each so nothing lazily built lands on a
    // measured one
    let _ = reader.read_column(id_column).expect("id");
    let _ = reader.read_column(blob_column).expect("blob");

    let (id_cell, _, narrow_bytes) = measure(|| {
        let decoded = reader.read_column(id_column).expect("id");
        decoded.cell(ROWS - 1).map(|c| c.len())
    });
    assert_eq!(id_cell, Some(8), "the id column decoded to real cells");
    let (blob_cell, _, wide_bytes) = measure(|| {
        let decoded = reader.read_column(blob_column).expect("blob");
        decoded.cell(ROWS - 1).map(|c| c.len())
    });
    assert_eq!(blob_cell, Some(BLOB_LEN));

    let payload = (ROWS * BLOB_LEN) as f64;
    assert!(
        wide_bytes as f64 > payload * 0.5,
        "the wide column should not have compressed away: {} bytes for a {} byte payload",
        wide_bytes,
        payload
    );
    assert!(
        narrow_bytes * 8 < wide_bytes,
        "reading the narrow column touched {} bytes against the wide column's {}",
        narrow_bytes,
        wide_bytes
    );
}

/// A file's own bounds are the union of its zones, so a file can admit a
/// constant that no zone holds. Rejecting it there costs the zone region
/// and leaves the payload unread, which is what makes the rows decoded
/// zero rather than merely fewer
#[test]
fn test_zone_map_prune_reduces_rows_decoded() {
    const ROWS: usize = 2_048;
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 15);
    let schema = schema(&[("id", TypeId::Int64)]);

    // Two zones of 1024 rows with a gap between them. Nothing in the file
    // is near 5000, but the file's bounds span it
    let ids: Vec<Option<Vec<u8>>> = (0..ROWS as i64)
        .map(|r| {
            let v = if (r as usize) < 1_024 { r } else { r + 8_000 };
            Some(v.to_le_bytes().to_vec())
        })
        .collect();
    let columns = vec![ColumnData {
        column_id: 0,
        cells: ids,
    }];
    zyron_lake::write_data_file(
        &paths,
        &schema,
        &WriteRequest {
            partition_id: 3,
            columns: &columns,
            sort_keys: &[0],
            sort_strategies: &[],
            cluster_spec_id: 0,
            table_id: 15,
            bloom_columns: &[],
            index_id: None,
        },
    )
    .expect("write data file");

    let reader = zyron_lake::LakeFileReader::open(&paths, 3).expect("open");
    let id_column = schema.column_by_id(0).expect("id column");

    let in_the_gap = StoredFilter::lower(&cmp(0, CompareOp::Eq, 5_000), &schema).expect("lowers");
    let in_a_zone = StoredFilter::lower(&cmp(0, CompareOp::Eq, 500), &schema).expect("lowers");

    let _ = reader.rows_matching(&in_the_gap).expect("filter");
    let _ = reader.read_column(id_column).expect("decode");

    let (pruned, _, prune_bytes) = measure(|| reader.rows_matching(&in_the_gap).expect("filter"));
    let pruned = pruned.expect("the filter decided");
    assert!(
        pruned.iter().all(|b| *b == 0),
        "no zone of this file can hold 5000"
    );

    let (_, _, decode_bytes) = measure(|| {
        let decoded = reader.read_column(id_column).expect("decode");
        decoded.cell(0).map(|c| c.len())
    });
    assert!(
        prune_bytes * 4 < decode_bytes,
        "rejecting from zone maps touched {} bytes against {} to decode the column",
        prune_bytes,
        decode_bytes
    );

    // The same predicate one zone does hold keeps exactly its rows
    let kept = reader
        .rows_matching(&in_a_zone)
        .expect("filter")
        .expect("decided");
    let surviving: Vec<usize> = (0..ROWS)
        .filter(|r| kept[r / 8] & (1 << (r % 8)) != 0)
        .collect();
    assert_eq!(surviving, vec![500], "one row holds 500");
}

/// A dictionary segment answers a term from its distinct values and its
/// codes, so a column whose decoded size dwarfs its encoded size never
/// materializes the difference. The mask still has to be the one a full
/// decode would have produced
#[test]
fn test_predicate_evaluated_on_encoded_data_without_full_decode() {
    const ROWS: usize = 1_024;
    const DISTINCT: usize = 8;
    const CELL_LEN: usize = 64;

    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 16);
    let schema = schema(&[("tag", TypeId::Varchar)]);

    let value_of = |k: usize| -> Vec<u8> {
        let mut cell = format!("tag-{:02}-", k).into_bytes();
        cell.resize(CELL_LEN, b'0' + (k as u8 % 10));
        cell
    };
    let tags: Vec<Option<Vec<u8>>> = (0..ROWS).map(|r| Some(value_of(r % DISTINCT))).collect();
    let columns = vec![ColumnData {
        column_id: 0,
        cells: tags,
    }];
    zyron_lake::write_data_file(
        &paths,
        &schema,
        &WriteRequest {
            partition_id: 4,
            columns: &columns,
            sort_keys: &[],
            sort_strategies: &[],
            cluster_spec_id: 0,
            table_id: 16,
            bloom_columns: &[],
            index_id: None,
        },
    )
    .expect("write data file");

    let reader = zyron_lake::LakeFileReader::open(&paths, 4).expect("open");
    let tag_column = schema.column_by_id(0).expect("tag column");
    let target = value_of(3);
    let filter = StoredFilter::lower(
        &LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Eq,
            value: LakeValue::Str(String::from_utf8(target.clone()).expect("utf8")),
        },
        &schema,
    )
    .expect("lowers");

    let _ = reader.rows_matching(&filter).expect("filter");
    let _ = reader.read_column(tag_column).expect("decode");

    let (mask, _, encoded_bytes) = measure(|| reader.rows_matching(&filter).expect("filter"));
    let mask = mask.expect("the filter decided");
    let (_, _, decode_bytes) = measure(|| {
        let decoded = reader.read_column(tag_column).expect("decode");
        decoded.cell(0).map(|c| c.len())
    });

    // The mask is exactly what comparing every decoded cell would give
    let decoded = reader.read_column(tag_column).expect("decode");
    let expected: Vec<usize> = (0..ROWS)
        .filter(|r| decoded.cell(*r) == Some(target.as_slice()))
        .collect();
    let surviving: Vec<usize> = (0..ROWS)
        .filter(|r| mask[r / 8] & (1 << (r % 8)) != 0)
        .collect();
    assert_eq!(surviving, expected, "the encoded answer is the decoded one");
    assert_eq!(surviving.len(), ROWS / DISTINCT);

    // The property is that the decoded column is never built, whichever
    // encoding the writer picked for this data. Both paths read the same
    // segment, so the difference between them is the decoded form, and it
    // has to account for the whole column
    let payload = ROWS * CELL_LEN;
    assert!(
        decode_bytes as f64 > payload as f64 * 0.5,
        "decoding should materialize the column: {} bytes for a {} byte payload",
        decode_bytes,
        payload
    );
    assert!(
        decode_bytes - encoded_bytes >= (payload as f64 * 0.8) as usize,
        "answering from the encoding should skip the decoded column: {} bytes against {}, \
         a saving of {} for a {} byte payload",
        encoded_bytes,
        decode_bytes,
        decode_bytes - encoded_bytes,
        payload
    );
    assert!(
        encoded_bytes < payload,
        "answering from the encoding touched {} bytes, more than the column decodes to",
        encoded_bytes
    );
}

fn attempt(operation: OperationKind) -> CommitAttempt<'static> {
    CommitAttempt {
        operation,
        db_txn_id: 0,
        commit_lsn: 1,
        timestamp_us: 1_754_700_000_000_000,
        read_predicate: None,
        audit: None,
    }
}

fn new_log(paths: &LakePaths, schema: &LakeSchema) -> TransactionLog {
    TransactionLog::create(
        paths.clone(),
        attempt(OperationKind::SchemaChange),
        schema,
        None,
        &BTreeMap::new(),
    )
    .expect("create log")
}

/// Observing a planned scan walks the predicate tree and moves atomic
/// counters. It runs once per planned scan on a worker thread that will
/// run many of them, and the observer's whole design is a fixed slot
/// array of epoch-tagged atomics precisely so it never allocates
#[test]
fn test_zero_allocation_in_workload_observer() {
    let predicates = predicate_catalog();
    // The epoch reads a clock, so it is taken outside the measured region
    let epoch = current_epoch();
    // One untimed pass so the process-global observer and its slot array
    // exist before anything is counted
    for p in &predicates {
        observe_scan(7, p, 4_096, 1_024, epoch);
        observe_scan_result(7, p, 1_000, 10, epoch);
    }

    for predicate in &predicates {
        let (_, allocs, bytes) = measure(|| {
            for _ in 0..64 {
                observe_scan(7, predicate, 4_096, 1_024, epoch);
                observe_scan_result(7, predicate, 1_000, 10, epoch);
            }
        });
        assert_eq!(
            (allocs, bytes),
            (0, 0),
            "observing {:?} allocated {} times for {} bytes",
            predicate,
            allocs,
            bytes
        );
    }
}

/// The gate replays every predicate class against every file before it
/// will accept a rewrite. That sweep is the expensive part and allocates
/// nothing, and the gate around it holds one score per class, so its own
/// allocation count has to stay flat as the file set grows
#[test]
fn test_zero_allocation_in_feedback_gate() {
    let manifest = synthetic_manifest(4_096);
    let probe = cmp(0, CompareOp::Lt, 100_000);

    let _ = skip_rate(&manifest.entries, &manifest.schema, &probe);
    let (rate, allocs, bytes) = measure(|| skip_rate(&manifest.entries, &manifest.schema, &probe));
    assert!(
        rate > 0.0 && rate < 1.0,
        "the probe has to prune something and keep something, got {}",
        rate
    );
    assert_eq!(
        (allocs, bytes),
        (0, 0),
        "replaying one class over {} files allocated {} times for {} bytes",
        manifest.entries.len(),
        allocs,
        bytes
    );

    let classes: Vec<PredicateClass> = predicate_catalog()
        .into_iter()
        .map(|predicate| PredicateClass {
            predicate,
            weight: 1.0,
            measured_skip_rate: None,
        })
        .collect();
    // No anchors and one free key, so the legality check passes without
    // building the vectors a conflict would report
    let anchors: [u32; 0] = [];
    let keys = [0u32];

    let gate_allocs = |files: usize| -> usize {
        let m = synthetic_manifest(files);
        let run = || {
            evaluate(
                &m.entries,
                &m.entries,
                &m.schema,
                &classes,
                &anchors,
                &keys,
                GateConfig::default(),
            )
        };
        let _ = run();
        let (_, allocs, _) = measure(run);
        allocs
    };

    let small = gate_allocs(100);
    let large = gate_allocs(10_000);
    assert_eq!(
        small, large,
        "the gate allocated {} times over 100 files and {} times over 10,000,          so its cost scales with the file set rather than the class count",
        small, large
    );
    assert!(
        small <= 1,
        "the gate should hold one score buffer, not {}",
        small
    );
}

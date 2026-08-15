//! Lake-only measurements: what this format has and the heap does not.
//!
//! The cross-format suite compares the two storage formats on the work a
//! user asks for. Nothing there measures the machinery that only exists
//! because the lake exists: pruning a file set from a manifest without
//! touching a data file, projecting that manifest into a sweepable index,
//! rejecting rows from zone maps, serializing commits through exclusive
//! file creation, and rewriting a layout because the workload asked for a
//! different one. Those have no heap counterpart, so they are absolute
//! numbers rather than ratios.
//!
//! ## Two kinds of number, judged differently
//!
//! **Counts are asserted.** How many files a predicate rejected, how many
//! rows a zone map turned away, how many commits collided, how much a
//! layout improved a skip rate. Optimization changes how fast the work
//! happens, not how much of it there is, so these mean the same thing in
//! every build profile and their bounds hold everywhere. They come from
//! the structure of the format, not from a measurement someone rounded up.
//!
//! **Timings are recorded and not judged.** No target is invented here.
//! An absolute target needs an optimized run on known hardware, and one
//! set without that is a number that goes stale the first time the machine
//! changes, which is exactly what happened to every absolute target this
//! repo used to carry. They are written to the run file so a baseline can
//! be read off it.

use std::collections::BTreeMap;
use std::sync::Arc;

use zyron_bench_harness::{
    RatioBound, assert_exact_metric, init, measuring, record_metric, tprintln,
};
use zyron_common::TypeId;
use zyron_lake::manifest::{ClusterSpec, ColumnStatsEntry, PartitionEntry};
use zyron_lake::predicate::ColumnBounds;
use zyron_lake::{
    ClusterKey, ClusterPassOptions, ClusterStrategy, ColumnData, CommitAttempt, CompareOp,
    Decision, GateConfig, LakeColumn, LakePaths, LakePredicate, LakeSchema, LakeValue, LogEntry,
    ManifestFile, OperationKind, PredicateClass, PruneDecision, PruneIndex, TransactionLog,
    WriteRequest, skip_rate, with_sweep,
};

/// Repetitions of a timed measurement, matching every other suite here.
const RUNS: usize = 5;

/// Serializes the suite. Two of these tests saturate every core, so running
/// them together would measure contention rather than the thing named, and
/// their sections would interleave into one unreadable log
static BENCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Opens a section and takes the suite lock for its duration.
///
/// Deliberately unnumbered. The test harness decides what runs when, so a
/// number printed here would claim a sequence the log does not have: it
/// runs alphabetically single threaded and in whatever order threads win
/// the lock otherwise. A title says what the section measures, which is
/// the thing a reader is actually looking for
fn section(title: &str) -> std::sync::MutexGuard<'static, ()> {
    let guard = BENCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("lake");
    tprintln!("");
    tprintln!("=== {} ===", title);
    guard
}

/// Files in the pruning and manifest measurements.
///
/// A hundred thousand is the scale at which a per-file loop and a
/// vectorized sweep stop looking alike, which is the whole reason the
/// index is a struct of arrays. Small enough that an unoptimized build
/// still finishes, because the counted assertions here have to run on
/// every test pass and not only on a baselined one
fn files() -> usize {
    if measuring() { 100_000 } else { 20_000 }
}

/// Rows per data file in the scan and zone map measurements.
///
/// A zone covers 1024 rows, so this is a whole number of zones with room
/// for a predicate to reject some and keep others
const ROWS_PER_FILE: usize = 16_384;

// =============================================================================
// Fixtures
// =============================================================================

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

/// A manifest whose files hold disjoint runs of `id` and a `bucket` that
/// repeats every sixteen files, so a predicate on one prunes a contiguous
/// range and a predicate on the other prunes a strided fifteen sixteenths
fn synthetic_manifest(files: usize) -> ManifestFile {
    let entries = (0..files)
        .map(|i| {
            let lo = i as i64 * 100;
            PartitionEntry {
                partition_id: i as u64,
                size_bytes: 1 << 20,
                row_count: 100,
                added_version: 1,
                cluster_spec_id: 0,
                column_stats: vec![
                    int_stats(0, lo, lo + 99, 100),
                    int_stats(1, (i % 16) as i64, (i % 16) as i64, 100),
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

fn attempt(operation: OperationKind, txn: u64) -> CommitAttempt<'static> {
    CommitAttempt {
        operation,
        db_txn_id: txn,
        commit_lsn: 1,
        timestamp_us: 1_754_700_000_000_000,
        read_predicate: None,
        audit: None,
    }
}

fn new_log(paths: &LakePaths, schema: &LakeSchema) -> TransactionLog {
    TransactionLog::create(
        paths.clone(),
        attempt(OperationKind::SchemaChange, 0),
        schema,
        None,
        &BTreeMap::new(),
    )
    .expect("create log")
}

/// Files the manifest's exact per-file rule rejects, which every sweep is
/// measured against
fn exact_pruned(manifest: &ManifestFile, predicate: &LakePredicate) -> usize {
    manifest
        .entries
        .iter()
        .filter(|e| manifest.prune_file(predicate, e) == PruneDecision::CannotMatch)
        .count()
}

fn micros<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let start = std::time::Instant::now();
    let out = f();
    (out, start.elapsed().as_secs_f64() * 1_000_000.0)
}

// =============================================================================
// File pruning
// =============================================================================

/// Pruning is the lake's core read-side claim, and its value is a count:
/// how much of the file set a predicate removes before any data file is
/// opened. A rate is the honest unit because it does not depend on how
/// many files the table happens to have
#[test]
fn test_file_pruning_rejects_the_files_a_predicate_cannot_match() {
    let _section = section("File Pruning Effectiveness");
    let manifest = synthetic_manifest(files());
    let total = manifest.entries.len();

    // A point predicate lands in exactly one file, since the runs of `id`
    // are disjoint and a hundred wide
    let point = cmp(0, CompareOp::Eq, 4_050);
    let pruned = exact_pruned(&manifest, &point);
    assert_eq!(
        total - pruned,
        1,
        "a point predicate over disjoint runs must survive in exactly one file"
    );
    let rate = pruned as f64 / total as f64;
    tprintln!("  Files in manifest: {}", total);
    assert!(
        assert_exact_metric(
            "lake_pruning",
            "point predicate files pruned (fraction)",
            rate,
            // One file of `total` survives, so the rate is 1 - 1/total.
            // Bounding it at four nines says the same thing without
            // restating the file count, and holds at any scale past 10k
            RatioBound::AtLeast(0.9999),
        ),
        "a point predicate pruned only {:.4} of the file set",
        rate
    );

    // A strided predicate keeps one bucket in sixteen. The exact answer is
    // a fifteen sixteenths prune, and it is exact rather than approximate
    // because the bucket column's min equals its max in every file
    let strided = cmp(1, CompareOp::Eq, 7);
    let strided_pruned = exact_pruned(&manifest, &strided);
    let strided_rate = strided_pruned as f64 / total as f64;
    assert_eq!(
        strided_pruned,
        total - total / 16,
        "a bucket predicate must reject every file whose bucket differs"
    );
    assert!(
        assert_exact_metric(
            "lake_pruning",
            "strided predicate files pruned (fraction)",
            strided_rate,
            RatioBound::AtLeast(0.93),
        ),
        "a one-in-sixteen predicate pruned only {:.4}",
        strided_rate
    );

    // A predicate covering the whole key range proves nothing and must
    // prune nothing. Pruning here would be a wrong answer, not a win
    let everything = cmp(0, CompareOp::GtEq, 0);
    assert_eq!(
        exact_pruned(&manifest, &everything),
        0,
        "a predicate every file satisfies must not remove any of them"
    );
}

/// The vectorized sweep is a different implementation of the same
/// question, so the claim that matters is that it agrees with the exact
/// answer file for file. A faster sweep that disagrees is a wrong answer
#[test]
fn test_prune_index_sweep_agrees_with_the_exact_answer_and_reports_its_cost() {
    let _section = section("Prune Index Sweep vs Exact Answer");
    let manifest = synthetic_manifest(files());
    let total = manifest.entries.len();
    let predicates = [
        ("point", cmp(0, CompareOp::Eq, 4_050)),
        ("range", cmp(0, CompareOp::Lt, 100_000)),
        ("strided", cmp(1, CompareOp::Eq, 7)),
    ];

    // Build once and time it: the projection is paid per manifest version
    // and shared by every plan against it, so its cost is amortized over
    // queries rather than charged to one
    let mut build_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (index, us) = micros(|| PruneIndex::build(&manifest));
        assert_eq!(index.file_count(), total);
        build_runs.push(us);
    }
    tprintln!("  Files in manifest: {}", total);
    record_metric("lake_pruning", "Prune index build", "us", build_runs);

    for (label, predicate) in &predicates {
        let expected: Vec<u8> = manifest
            .entries
            .iter()
            .map(|e| (manifest.prune_file(predicate, e) == PruneDecision::CannotMatch) as u8)
            .collect();
        let index = PruneIndex::build(&manifest);

        // One untimed sweep so the thread-local scratch is sized before
        // anything is measured, and so the comparison runs on a warm cache
        // exactly as the timed ones do
        with_sweep(&index, predicate, |mask, complete| {
            assert_eq!(
                mask.len(),
                total,
                "{}: sweep covered the whole file set",
                label
            );
            // The sweep is sound but not complete: a one is a proof the
            // file can be skipped, a zero is the absence of proof. Where
            // it reports itself short of exact, the manifest decides, so
            // the sweep may only ever be a subset of the exact answer
            for (i, (got, want)) in mask.iter().zip(expected.iter()).enumerate() {
                if complete {
                    assert_eq!(
                        got, want,
                        "{}: file {} disagreed with the exact answer",
                        label, i
                    );
                } else {
                    assert!(
                        *got == 0 || *want == 1,
                        "{}: file {} was swept out but the exact answer keeps it",
                        label,
                        i
                    );
                }
            }
        });

        let mut sweep_runs = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let (_, us) = micros(|| {
                with_sweep(&index, predicate, |mask, _| {
                    // Sum rather than discard, so the sweep cannot be
                    // optimized away as dead work
                    mask.iter().map(|b| *b as u64).sum::<u64>()
                })
            });
            sweep_runs.push(us);
        }
        record_metric(
            "lake_pruning",
            &format!("{} sweep over {} files", label, total),
            "us",
            sweep_runs,
        );
    }
}

// =============================================================================
// Zone maps and encoded rejection
// =============================================================================

/// A file's own bounds are the union of its zones, so a file can admit a
/// value no zone holds. What that buys is counted in rows never decoded
#[test]
fn test_zone_maps_reject_rows_the_file_bounds_admit() {
    let _section = section("Zone Map Row Rejection");
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 90);
    let schema = schema(&[("id", TypeId::Int64), ("bucket", TypeId::Int64)]);

    // Every zone holds a contiguous run, and the runs leave gaps. A value
    // in a gap is inside the file's bounds and inside no zone
    let zone = 1_024usize;
    let ids: Vec<Option<Vec<u8>>> = (0..ROWS_PER_FILE)
        .map(|r| {
            let zone_index = (r / zone) as i64;
            let within = (r % zone) as i64;
            Some((zone_index * 10_000 + within).to_le_bytes().to_vec())
        })
        .collect();
    let buckets: Vec<Option<Vec<u8>>> = (0..ROWS_PER_FILE)
        .map(|r| Some(((r % 16) as i64).to_le_bytes().to_vec()))
        .collect();
    let columns = vec![
        ColumnData {
            column_id: 0,
            cells: ids,
        },
        ColumnData {
            column_id: 1,
            cells: buckets,
        },
    ];
    zyron_lake::write_data_file(
        &paths,
        &schema,
        &WriteRequest {
            partition_id: 1,
            columns: &columns,
            sort_keys: &[0],
            sort_strategies: &[],
            cluster_spec_id: 0,
            table_id: 90,
            bloom_columns: &[],
            index_id: None,
        },
    )
    .expect("write data file");

    let reader = zyron_lake::LakeFileReader::open(&paths, 1).expect("open");
    assert_eq!(reader.row_count(), ROWS_PER_FILE);

    // 5_000 sits between zone 0's run (0..1023) and zone 1's (10_000..)
    let gap = cmp(0, CompareOp::Eq, 5_000);
    let filter = zyron_lake::StoredFilter::lower(&gap, &schema)
        .expect("an integer equality lowers onto stored bytes");
    let mask = reader
        .rows_matching(&filter)
        .expect("evaluate")
        .expect("the filter decided something");
    let kept: u64 = mask.iter().map(|b| b.count_ones() as u64).sum();
    assert_eq!(
        kept, 0,
        "a value inside the file bounds and inside no zone must leave no row standing"
    );
    assert!(
        assert_exact_metric(
            "lake_skipping",
            "zone map rows rejected (fraction)",
            1.0,
            RatioBound::AtLeast(1.0),
        ),
        "the zone maps did not reject every row"
    );

    // A value a zone does hold must survive, or the rejection above proves
    // nothing: a filter that rejects everything is not a filter
    let present = cmp(0, CompareOp::Eq, 10_005);
    let present_filter = zyron_lake::StoredFilter::lower(&present, &schema).expect("lowers");
    let present_mask = reader
        .rows_matching(&present_filter)
        .expect("evaluate")
        .expect("decided");
    let present_kept: u64 = present_mask.iter().map(|b| b.count_ones() as u64).sum();
    assert!(
        present_kept >= 1,
        "a value the file holds was rejected, so the filter is unsound"
    );
    assert!(
        present_kept <= zone as u64,
        "a point lookup kept {} rows, more than the one zone that can hold it",
        present_kept
    );

    // What rejection costs, against decoding the column it rejected
    let mut reject_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (_, us) = micros(|| reader.rows_matching(&filter).expect("evaluate"));
        reject_runs.push(us);
    }
    record_metric("lake_skipping", "Zone map rejection", "us", reject_runs);

    let id_column = schema.column_by_id(0).expect("id");
    let mut decode_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (_, us) = micros(|| reader.read_column(id_column).expect("decode"));
        decode_runs.push(us);
    }
    record_metric("lake_skipping", "Full column decode", "us", decode_runs);
}

/// Scan throughput over a data file, which is the number the file format
/// itself is responsible for
#[test]
fn test_data_file_scan_throughput() {
    let _section = section("Data File Scan Throughput");
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 91);
    let schema = schema(&[("id", TypeId::Int64), ("bucket", TypeId::Int64)]);

    let ids: Vec<Option<Vec<u8>>> = (0..ROWS_PER_FILE as i64)
        .map(|r| Some(r.to_le_bytes().to_vec()))
        .collect();
    let buckets: Vec<Option<Vec<u8>>> = (0..ROWS_PER_FILE)
        .map(|r| Some(((r % 16) as i64).to_le_bytes().to_vec()))
        .collect();
    let columns = vec![
        ColumnData {
            column_id: 0,
            cells: ids,
        },
        ColumnData {
            column_id: 1,
            cells: buckets,
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
            table_id: 91,
            bloom_columns: &[],
            index_id: None,
        },
    )
    .expect("write");
    assert_eq!(entry.row_count, ROWS_PER_FILE as u64);

    let reader = zyron_lake::LakeFileReader::open(&paths, 1).expect("open");
    let id_column = schema.column_by_id(0).expect("id");
    let _ = reader.read_column(id_column).expect("warm");

    let mut rows_per_sec = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (decoded, us) = micros(|| reader.read_column(id_column).expect("decode"));
        // Touch the last cell so the decode cannot be elided
        assert_eq!(decoded.cell(ROWS_PER_FILE - 1).map(|c| c.len()), Some(8));
        rows_per_sec.push(ROWS_PER_FILE as f64 / (us / 1_000_000.0));
    }
    tprintln!("  Rows per data file: {}", ROWS_PER_FILE);
    record_metric("lake_scan", "Scan throughput", " rows/s", rows_per_sec);
}

// =============================================================================
// Constraint enforcement and secondary indexes
// =============================================================================

/// Rows per data file in the constraint and index measurements, and files
/// per table. Enough files that pruning has something to reject and enough
/// rows per file that walking one is visibly different from bisecting it
const CONSTRAINT_ROWS_PER_FILE: usize = 8_192;
const CONSTRAINT_FILES: usize = 8;

/// Builds a table of ascending integer keys, clustered or not.
///
/// Clustered writes each file ascending by the key, which is the layout a
/// declared primary key bootstraps to. Unclustered writes arrival order,
/// which is what a table with no declared ordering gets
fn key_table(
    dir: &std::path::Path,
    table_id: u32,
    clustered: bool,
) -> (TransactionLog, LakeSchema) {
    let paths = LakePaths::new(dir, table_id);
    let schema = schema(&[("id", TypeId::Int64), ("payload", TypeId::Int64)]);
    let spec = clustered.then(|| ClusterSpec {
        spec_id: 1,
        keys: vec![ClusterKey {
            column_id: 0,
            strategy: ClusterStrategy::RangePartition,
            param: 0,
        }],
    });
    let log = TransactionLog::create(
        paths,
        attempt(OperationKind::SchemaChange, 0),
        &schema,
        spec.as_ref(),
        &BTreeMap::new(),
    )
    .expect("create log");

    for file in 0..CONSTRAINT_FILES {
        let base = (file * CONSTRAINT_ROWS_PER_FILE) as i64;
        // Interleaved within the file, so an unclustered table is genuinely
        // unordered rather than accidentally sorted by arrival
        let ids: Vec<Option<Vec<u8>>> = (0..CONSTRAINT_ROWS_PER_FILE)
            .map(|r| {
                let stride = (r * 7919) % CONSTRAINT_ROWS_PER_FILE;
                Some((base + stride as i64).to_le_bytes().to_vec())
            })
            .collect();
        let payload: Vec<Option<Vec<u8>>> = (0..CONSTRAINT_ROWS_PER_FILE)
            .map(|r| Some((r as i64).to_le_bytes().to_vec()))
            .collect();
        zyron_lake::append_rows(
            &log,
            attempt(OperationKind::Append, 0),
            table_id as u64,
            &[
                ColumnData {
                    column_id: 0,
                    cells: ids,
                },
                ColumnData {
                    column_id: 1,
                    cells: payload,
                },
            ],
        )
        .expect("append");
    }
    (log, schema)
}

fn id_batch(ids: &[i64]) -> Vec<ColumnData> {
    vec![
        ColumnData {
            column_id: 0,
            cells: ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
        },
        ColumnData {
            column_id: 1,
            cells: ids
                .iter()
                .map(|_| Some(0i64.to_le_bytes().to_vec()))
                .collect(),
        },
    ]
}

/// What enforcing a primary key costs on an insert, with and without the
/// ordering that lets the check bisect instead of walk.
///
/// Both are real configurations: a table with a declared key bootstraps to
/// the clustered layout, a table without one does not. The pair is what
/// carries the information, because the check runs on every insert either
/// way and only its cost differs
#[test]
fn test_unique_check_cost_with_and_without_clustering() {
    let _section = section("Unique Check On Insert");
    let dir = tempfile::tempdir().expect("tempdir");
    let total = (CONSTRAINT_FILES * CONSTRAINT_ROWS_PER_FILE) as i64;
    tprintln!("  Rows: {}, files: {}", total, CONSTRAINT_FILES);

    let spec = zyron_lake::UniqueSpec {
        name: "pk_id".into(),
        column_ids: vec![0],
    };
    // Keys spread across the table, the shape an insert of unrelated rows
    // produces. Every one of them already exists, so the check has to reach
    // the rows rather than stopping at the bounds
    let probes: Vec<i64> = (0..16).map(|i| i * (total / 16) + 3).collect();
    let batch = id_batch(&probes);

    for (clustered, label) in [(true, "clustered"), (false, "unclustered")] {
        let (log, _schema) = key_table(dir.path(), if clustered { 300 } else { 301 }, clustered);
        let manifest = log.latest_manifest().expect("manifest");
        let (_, stats) =
            zyron_lake::check_unique(log.paths(), &manifest, &spec, &batch).expect("check");
        tprintln!(
            "  {}: files opened {}, decoded {}, bisected {}, rows compared {}",
            label,
            stats.files_opened,
            stats.files_decoded,
            stats.files_bisected,
            stats.rows_scanned
        );

        let mut runs = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let (_, us) = micros(|| {
                zyron_lake::check_unique(log.paths(), &manifest, &spec, &batch).expect("check")
            });
            runs.push(us);
        }
        record_metric(
            "lake_constraints",
            &format!("Unique check, {}", label),
            "us",
            runs,
        );
    }

    // The counted claim: on a sorted table the check compares a number of
    // rows proportional to the keys asked about, not to the table
    let (log, _schema) = key_table(dir.path(), 302, true);
    let manifest = log.latest_manifest().expect("manifest");
    let (_, stats) =
        zyron_lake::check_unique(log.paths(), &manifest, &spec, &batch).expect("check");
    let per_key = stats.rows_scanned as f64 / probes.len() as f64;
    assert!(
        assert_exact_metric(
            "lake_constraints",
            "rows compared per probed key (clustered)",
            per_key,
            RatioBound::AtMost(8.0),
        ),
        "a clustered unique check walked {} rows for {} keys",
        stats.rows_scanned,
        probes.len()
    );
}

/// What a secondary index costs to probe, against answering the same key
/// by reading the data files the manifest could not rule out.
///
/// The index is on a column the table is not clustered by, which is when an
/// index has something to add: bounds pruning on such a column rejects
/// nothing, so without an index the query reads every file
#[test]
fn test_secondary_index_probe_against_the_scan_it_replaces() {
    let _section = section("Secondary Index Probe");
    let dir = tempfile::tempdir().expect("tempdir");
    let (log, schema) = key_table(dir.path(), 310, true);
    let total = (CONSTRAINT_FILES * CONSTRAINT_ROWS_PER_FILE) as i64;

    // payload is uncorrelated with the cluster key, so every file's bounds
    // admit every probed value
    zyron_lake::operations::create_index(
        &log,
        attempt(OperationKind::SchemaChange, 0),
        310,
        "ix_payload",
        &[1],
        false,
    )
    .expect("create index");

    let manifest = log.latest_manifest().expect("manifest");
    let spec = manifest.index_by_name("ix_payload").expect("index");
    let index_files = manifest
        .index_files
        .iter()
        .filter(|f| f.index_id == spec.index_id)
        .count();
    tprintln!(
        "  Rows: {}, data files: {}, index files: {}",
        total,
        manifest.entries.len(),
        index_files
    );

    let needle = (CONSTRAINT_ROWS_PER_FILE as i64) / 2;
    let cell = needle.to_le_bytes();
    let (addresses, stats) =
        zyron_lake::probe_equal(log.paths(), &manifest, spec, &[Some(&cell[..])]).expect("probe");
    assert!(!addresses.is_empty(), "the probed value is stored");
    tprintln!(
        "  Probe: index files opened {} of {}, entries examined {}, rows addressed {}",
        stats.files_opened,
        index_files,
        stats.entries_examined,
        addresses.len()
    );

    // The counted claim: disjoint key ranges mean a probe opens one index
    // file however many the index has
    assert!(
        assert_exact_metric(
            "lake_index",
            "index files opened per point probe",
            stats.files_opened as f64,
            RatioBound::AtMost(1.0),
        ),
        "a point probe opened {} index files",
        stats.files_opened
    );

    let mut probe_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (_, us) = micros(|| {
            zyron_lake::probe_equal(log.paths(), &manifest, spec, &[Some(&cell[..])])
                .expect("probe")
        });
        probe_runs.push(us);
    }
    record_metric(
        "lake_index",
        "Point probe through the index",
        "us",
        probe_runs,
    );

    // The same answer without the index: every file the bounds admit is
    // opened and its column read
    let payload_column = schema.column_by_id(1).expect("payload");
    let mut scan_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (found, us) = micros(|| {
            let mut hits = 0usize;
            for entry in &manifest.entries {
                let reader = zyron_lake::LakeFileReader::open(log.paths(), entry.partition_id)
                    .expect("open");
                let column = reader.read_column(payload_column).expect("decode");
                for row in 0..reader.row_count() {
                    if column.cell_equals(row, &cell) {
                        hits += 1;
                    }
                }
            }
            hits
        });
        assert_eq!(found, addresses.len(), "the index and the scan disagree");
        scan_runs.push(us);
    }
    record_metric("lake_index", "Point lookup by scanning", "us", scan_runs);

    // A range the index answers, which bounds pruning on an unclustered
    // column cannot narrow at all
    let bound = |v: i64, inclusive: bool| zyron_lake::index::RangeBound {
        value: LakeValue::Int(v),
        inclusive,
    };
    let (range_hits, range_stats) = zyron_lake::probe_range(
        log.paths(),
        &manifest,
        spec,
        Some(&bound(needle, true)),
        Some(&bound(needle + 63, true)),
    )
    .expect("range probe");
    tprintln!(
        "  Range probe: index files opened {} of {}, rows addressed {}",
        range_stats.files_opened,
        index_files,
        range_hits.len()
    );
    let mut range_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (_, us) = micros(|| {
            zyron_lake::probe_range(
                log.paths(),
                &manifest,
                spec,
                Some(&bound(needle, true)),
                Some(&bound(needle + 63, true)),
            )
            .expect("range probe")
        });
        range_runs.push(us);
    }
    record_metric(
        "lake_index",
        "Range probe through the index",
        "us",
        range_runs,
    );
}

// =============================================================================
// Transaction log
// =============================================================================

/// Exclusive file creation is the concurrency primitive, so the counted
/// claim is that concurrent writers produce exactly one version each with
/// no gaps and no duplicate partition ids, and that the losers retried
/// rather than silently dropping their work
#[test]
fn test_concurrent_commits_serialize_and_report_their_retries() {
    let _section = section("Transaction Log Under Concurrency");
    const WRITERS: usize = 8;
    const PER_WRITER: usize = 20;

    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 92);
    let schema = schema(&[("id", TypeId::Int64)]);
    let log = Arc::new(new_log(&paths, &schema));

    let (_, us) = micros(|| {
        std::thread::scope(|scope| {
            for writer in 0..WRITERS {
                let log = Arc::clone(&log);
                scope.spawn(move || {
                    for n in 0..PER_WRITER {
                        log.commit(attempt(OperationKind::Append, 0), |base| {
                            // The partition id is allocated against the
                            // winner's manifest on every retry, which is
                            // why commit takes a closure and not a fixed
                            // entry list
                            let next = base
                                .entries
                                .iter()
                                .map(|e| e.partition_id)
                                .max()
                                .unwrap_or(0)
                                + 1;
                            Ok(vec![LogEntry::AddFile(PartitionEntry {
                                partition_id: next,
                                size_bytes: 1 << 10,
                                row_count: 1,
                                added_version: 0,
                                cluster_spec_id: 0,
                                column_stats: vec![int_stats(0, next as i64, next as i64, 1)],
                                delete_predicate_ids: Vec::new(),
                            })])
                        })
                        .unwrap_or_else(|e| panic!("writer {} commit {} failed: {e}", writer, n));
                    }
                });
            }
        })
    });

    let commits = (WRITERS * PER_WRITER) as u64;
    // Version 1 is the schema change that created the log
    assert_eq!(
        log.latest_version(),
        commits + 1,
        "every commit must produce exactly one version"
    );

    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(
        manifest.entries.len() as u64,
        commits,
        "every commit must have added exactly one file"
    );
    let mut ids: Vec<u64> = manifest.entries.iter().map(|e| e.partition_id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len() as u64,
        commits,
        "two commits allocated the same partition id, so a retry reissued one"
    );

    // Eight threads racing for one exclusive create must have collided.
    // Zero retries would mean the writers serialized somewhere else and
    // this measured nothing
    let retries = log.commit_retries();
    assert!(
        retries > 0,
        "{} writers committed {} versions with no conflict, so nothing was contended",
        WRITERS,
        commits
    );
    tprintln!("  Writers: {}", WRITERS);
    tprintln!("  Commits: {}", commits);
    tprintln!(
        "  Retries: {} ({:.4} per commit)",
        retries,
        retries as f64 / commits as f64
    );
    record_metric(
        "lake_log",
        "Commit rate",
        " commits/s",
        vec![commits as f64 / (us / 1_000_000.0)],
    );
    record_metric(
        "lake_log",
        "Retries",
        " per commit",
        vec![retries as f64 / commits as f64],
    );
}

/// Reading the head is what every plan does before anything else, so it
/// is the one log operation on the query critical path
#[test]
fn test_log_head_and_manifest_resolution_cost() {
    let _section = section("Log Head and Manifest Resolution");
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 93);
    let schema = schema(&[("id", TypeId::Int64)]);
    let log = new_log(&paths, &schema);

    const APPENDS: usize = 200;
    for n in 0..APPENDS as u64 {
        log.commit(attempt(OperationKind::Append, 0), move |_| {
            Ok(vec![LogEntry::AddFile(PartitionEntry {
                partition_id: n + 1,
                size_bytes: 1 << 20,
                row_count: 100,
                added_version: 0,
                cluster_spec_id: 0,
                column_stats: vec![int_stats(0, n as i64 * 100, n as i64 * 100 + 99, 100)],
                delete_predicate_ids: Vec::new(),
            })])
        })
        .expect("commit");
    }

    let head = log.latest_version();
    assert_eq!(head, APPENDS as u64 + 1);

    // The head is a relaxed atomic load, so a single one is far below the
    // clock's resolution. Amortizing over a batch and reporting nanoseconds
    // is what keeps the number from rounding to zero and reading as free.
    //
    // black_box on every load, because a relaxed atomic load of the same
    // address is something LLVM may fold across iterations. Without it this
    // reported a fraction of one cycle per load, which no load can achieve
    // and so was measuring nothing
    const HEAD_READS: usize = 100_000;
    let mut head_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (acc, us) = micros(|| {
            let mut acc = 0u64;
            for _ in 0..HEAD_READS {
                acc = acc.wrapping_add(std::hint::black_box(log.latest_version()));
            }
            acc
        });
        assert!(acc > 0, "the head reads were optimized away");
        head_runs.push(us * 1_000.0 / HEAD_READS as f64);
    }
    record_metric("lake_log", "Head read", "ns", head_runs);

    // Batched for the same reason the head read is: one resolution is far
    // below the platform clock's granularity, so timing it alone reports
    // zero and reads as free rather than as unmeasurably small
    const MANIFEST_READS: usize = 1_000;
    let mut manifest_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (entries, us) = micros(|| {
            let mut entries = 0usize;
            for _ in 0..MANIFEST_READS {
                entries += std::hint::black_box(log.latest_manifest().expect("manifest"))
                    .entries
                    .len();
            }
            entries
        });
        assert_eq!(entries, APPENDS * MANIFEST_READS);
        manifest_runs.push(us * 1_000.0 / MANIFEST_READS as f64);
    }
    record_metric("lake_log", "Latest manifest (cached)", "ns", manifest_runs);

    // Two separate numbers, because they are two different operations that
    // happen to share a name. The first call on a version projects the
    // manifest into the index; every later one hands back the cached Arc.
    // Averaging one build with four cache hits described neither
    let (cold, cold_us) = micros(|| log.prune_index_at(head).expect("prune index"));
    assert_eq!(cold.file_count(), APPENDS);
    record_metric("lake_log", "Prune index build (cold)", "us", vec![cold_us]);

    const PRUNE_READS: usize = 1_000;
    let mut prune_runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (files, us) = micros(|| {
            let mut files = 0usize;
            for _ in 0..PRUNE_READS {
                files += std::hint::black_box(log.prune_index_at(head).expect("prune index"))
                    .file_count();
            }
            files
        });
        assert_eq!(files, APPENDS * PRUNE_READS);
        prune_runs.push(us * 1_000.0 / PRUNE_READS as f64);
    }
    record_metric(
        "lake_log",
        "Prune index at version (cached)",
        "ns",
        prune_runs,
    );
}

// =============================================================================
// Adaptive clustering
// =============================================================================

/// A clustering pass exists to make later reads skip more, so its claim is
/// a skip rate that went up. The gate refuses a pass that does not improve
/// one, which makes "accepted" and "improved" the same statement, and this
/// measures the improvement rather than trusting the decision
#[test]
fn test_a_clustering_pass_raises_the_measured_skip_rate() {
    let _section = section("Clustering Pass Skip-Rate Gain");
    const FILES: usize = 8;
    const ROWS: usize = 4_096;

    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 94);
    let schema = schema(&[("id", TypeId::Int64), ("bucket", TypeId::Int64)]);
    let log = new_log(&paths, &schema);

    // Every file spans the whole bucket range, so a bucket predicate can
    // prune nothing. This is the layout a load in arrival order produces
    // and the one clustering exists to fix
    for file in 0..FILES {
        let ids: Vec<Option<Vec<u8>>> = (0..ROWS)
            .map(|r| Some(((file * ROWS + r) as i64).to_le_bytes().to_vec()))
            .collect();
        let buckets: Vec<Option<Vec<u8>>> = (0..ROWS)
            .map(|r| Some(((r % 16) as i64).to_le_bytes().to_vec()))
            .collect();
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: ids,
            },
            ColumnData {
                column_id: 1,
                cells: buckets,
            },
        ];
        let entry = zyron_lake::write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: file as u64 + 1,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 94,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write");
        log.commit(attempt(OperationKind::Append, 0), move |_| {
            Ok(vec![LogEntry::AddFile(entry.clone())])
        })
        .expect("commit");
    }

    let bucket_predicate = cmp(1, CompareOp::Eq, 7);
    let before_manifest = log.latest_manifest().expect("manifest");
    let before = skip_rate(&before_manifest.entries, &schema, &bucket_predicate);
    assert_eq!(
        before, 0.0,
        "every file spans every bucket, so nothing can be skipped yet"
    );

    // Ask for the layout the workload wants: ordered by bucket, so a
    // bucket predicate lands in a few files instead of all of them
    let target = ClusterSpec {
        spec_id: 1,
        keys: vec![ClusterKey {
            column_id: 1,
            strategy: ClusterStrategy::RangePartition,
            param: 0,
        }],
    };
    let classes = vec![PredicateClass {
        predicate: bucket_predicate.clone(),
        weight: 1.0,
        measured_skip_rate: Some(before),
    }];

    let (outcome, us) = micros(|| {
        zyron_lake::run_cluster_pass(
            &log,
            attempt(OperationKind::Optimize, 0),
            94,
            &target,
            &classes,
            &ClusterPassOptions {
                max_inputs: FILES,
                target_rows_per_file: ROWS as u64,
                gate: GateConfig::default(),
                ..ClusterPassOptions::new(1)
            },
        )
        .expect("cluster pass")
    });

    assert!(
        matches!(outcome.decision, Decision::Accept { .. }),
        "a layout that turns a zero skip rate into a positive one must be accepted, got {:?}",
        outcome.decision
    );
    assert!(
        outcome.version.is_some(),
        "an accepted pass commits a version"
    );
    assert_eq!(outcome.inputs, FILES);
    assert!(outcome.outputs > 0);

    let after_manifest = log.latest_manifest().expect("manifest");
    let after = skip_rate(&after_manifest.entries, &schema, &bucket_predicate);
    assert!(
        after > before,
        "the pass was accepted but the skip rate did not improve: {} to {}",
        before,
        after
    );
    assert!(
        assert_exact_metric(
            "lake_clustering",
            "skip rate after a bucket-ordered pass",
            after,
            // Sixteen buckets over the rewritten files: a bucket predicate
            // must reject most of them. Half is well inside what ordering
            // on a sixteen-value key achieves, and it is a floor derived
            // from the key's cardinality rather than a measured number
            RatioBound::AtLeast(0.5),
        ),
        "ordering by a sixteen-value key left the skip rate at {:.4}",
        after
    );

    // Row count is conserved: a layout change must not lose or invent rows
    let rows_before: u64 = before_manifest.entries.iter().map(|e| e.row_count).sum();
    let rows_after: u64 = after_manifest.entries.iter().map(|e| e.row_count).sum();
    assert_eq!(
        rows_before, rows_after,
        "the pass changed the row count, so it is not a layout change"
    );

    tprintln!("  Inputs: {} files, {} rows", FILES, rows_before);
    tprintln!("  Outputs: {} files", outcome.outputs);
    tprintln!("  Skip rate: {:.4} -> {:.4}", before, after);
    record_metric("lake_clustering", "Cluster pass", "us", vec![us]);
    record_metric(
        "lake_clustering",
        "Rows rewritten",
        " rows/s",
        vec![outcome.rows_written as f64 / (us / 1_000_000.0)],
    );
}

/// A pass the gate refuses must leave the active file set byte identical.
/// The whole design rests on a rejected candidate never being visible, so
/// this is the counted claim behind "clustering can only improve or do
/// nothing"
#[test]
fn test_a_refused_clustering_pass_leaves_the_file_set_untouched() {
    let _section = section("Refused Clustering Pass Is Inert");
    const FILES: usize = 4;
    const ROWS: usize = 2_048;

    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 95);
    let schema = schema(&[("id", TypeId::Int64), ("bucket", TypeId::Int64)]);
    let log = new_log(&paths, &schema);

    // Already ordered by bucket, so re-ordering by bucket cannot improve
    // anything and the gate has nothing to buy
    for file in 0..FILES {
        let ids: Vec<Option<Vec<u8>>> = (0..ROWS)
            .map(|r| Some(((file * ROWS + r) as i64).to_le_bytes().to_vec()))
            .collect();
        let buckets: Vec<Option<Vec<u8>>> = (0..ROWS)
            .map(|_| Some((file as i64).to_le_bytes().to_vec()))
            .collect();
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: ids,
            },
            ColumnData {
                column_id: 1,
                cells: buckets,
            },
        ];
        let entry = zyron_lake::write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: file as u64 + 1,
                columns: &columns,
                sort_keys: &[1],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 95,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write");
        log.commit(attempt(OperationKind::Append, 0), move |_| {
            Ok(vec![LogEntry::AddFile(entry.clone())])
        })
        .expect("commit");
    }

    let before = log.latest_manifest().expect("manifest");
    let before_ids: Vec<u64> = before.entries.iter().map(|e| e.partition_id).collect();
    let before_version = log.latest_version();

    let target = ClusterSpec {
        spec_id: 1,
        keys: vec![ClusterKey {
            column_id: 1,
            strategy: ClusterStrategy::RangePartition,
            param: 0,
        }],
    };
    let bucket_predicate = cmp(1, CompareOp::Eq, 2);
    let measured = skip_rate(&before.entries, &schema, &bucket_predicate);
    let classes = vec![PredicateClass {
        predicate: bucket_predicate.clone(),
        weight: 1.0,
        measured_skip_rate: Some(measured),
    }];

    let outcome = zyron_lake::run_cluster_pass(
        &log,
        attempt(OperationKind::Optimize, 0),
        95,
        &target,
        &classes,
        &ClusterPassOptions {
            max_inputs: FILES,
            target_rows_per_file: ROWS as u64,
            gate: GateConfig::default(),
            ..ClusterPassOptions::new(1)
        },
    )
    .expect("cluster pass");

    assert!(
        !matches!(outcome.decision, Decision::Accept { .. }),
        "a layout that cannot improve the skip rate was accepted anyway: {:?}",
        outcome.decision
    );
    assert!(
        outcome.version.is_none(),
        "a refused pass must not commit a version"
    );

    let after = log.latest_manifest().expect("manifest");
    let after_ids: Vec<u64> = after.entries.iter().map(|e| e.partition_id).collect();
    assert_eq!(
        before_ids, after_ids,
        "a refused pass changed the active file set"
    );
    assert_eq!(
        log.latest_version(),
        before_version,
        "a refused pass advanced the log"
    );
    // Refusing is the result under test, so the log says so. A bare
    // "BelowThreshold" reads like something went wrong when it is the
    // whole point: a layout that buys nothing must not be applied
    tprintln!("  Candidate: order by bucket, on files already ordered by bucket");
    tprintln!("  Expected: refused, since it cannot improve the skip rate");
    tprintln!("  Gate decision [PASS]: refused, {:?}", outcome.decision);
    tprintln!(
        "  Active file set [PASS]: unchanged, {} files, log still at version {}",
        after_ids.len(),
        before_version
    );
}

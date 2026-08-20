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
    RatioBound, assert_exact_metric, check_performance, init, measuring, record_metric, tprintln,
};
use zyron_common::TypeId;
use zyron_lake::manifest::{ClusterSpec, ColumnStatsEntry, PartitionEntry};
use zyron_lake::predicate::ColumnBounds;
use zyron_lake::{
    AllCommitted, ClusterKey, ClusterPassOptions, ClusterStrategy, ColumnData, CommitAttempt,
    CompareOp, Decision, GateConfig, LakeColumn, LakePaths, LakePredicate, LakeSchema, LakeValue,
    LogEntry, ManifestFile, OperationKind, PredicateClass, PruneDecision, PruneIndex,
    TransactionLog, WriteRequest, skip_rate, with_sweep,
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
                column_stats: std::sync::Arc::new(vec![
                    int_stats(0, lo, lo + 99, 100),
                    int_stats(1, (i % 16) as i64, (i % 16) as i64, 100),
                ]),
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
        read_version: 0,
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
                                column_stats: std::sync::Arc::new(vec![int_stats(
                                    0,
                                    next as i64,
                                    next as i64,
                                    1,
                                )]),
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
                column_stats: std::sync::Arc::new(vec![int_stats(
                    0,
                    n as i64 * 100,
                    n as i64 * 100 + 99,
                    100,
                )]),
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

// =============================================================================
// Phase 18 absolute targets
// =============================================================================
//
// Everything above measures a count or records a timing without judging it.
// This section is the other kind: fourteen absolute latency and throughput
// targets that decide whether the format is fit to ship.
//
// They are absolute rather than ratios, which is a real cost: an absolute
// number is a claim about one machine and goes stale when the machine
// changes. They are stated anyway because a format's commit latency is a
// promise to whoever is waiting on it, and a promise you can only express
// relative to yesterday's build is not one a user can plan against.
//
// The harness applies them only in a measuring build and reports them
// unchecked elsewhere, so an unoptimized run is never failed against
// optimized numbers.

/// Versions the server lets a log accumulate before collapsing it into a
/// manifest checkpoint.
///
/// The maintenance loop does this on every cycle. A benchmark that skipped
/// it would measure a log no running server ever has: reconstructing any
/// version would replay from the beginning of time, and both the manifest
/// load and the time travel numbers would be reporting the absence of
/// maintenance rather than the cost of the format
const CHECKPOINT_EVERY: u64 = 64;

/// Collapses a log the way the maintenance loop does, so what is measured
/// afterwards is a log in the shape a running server keeps it in
fn checkpoint_like_the_server(log: &TransactionLog) {
    let head = log.head_version();
    let mut at = CHECKPOINT_EVERY;
    while at <= head {
        log.checkpoint(at).expect("checkpoint");
        at += CHECKPOINT_EVERY;
    }
}

/// Rows one commit carries in the commit-latency gates.
///
/// A commit's cost is dominated by manifest serialization and the fsync,
/// not by the rows, so this is sized to be a realistic statement rather
/// than to stress the writer
fn commit_rows() -> usize {
    if measuring() { 10_000 } else { 500 }
}

/// Files the OPTIMIZE gate compacts. The target names a thousand
fn optimize_files() -> usize {
    if measuring() { 1_000 } else { 20 }
}

/// Rows the clustering pass gate carries. The target names ten million
fn cluster_pass_rows() -> usize {
    if measuring() { 10_000_000 } else { 20_000 }
}

/// Files the manifest-load gate reconstructs. The target names ten
/// thousand
fn manifest_files() -> usize {
    if measuring() { 10_000 } else { 500 }
}

/// Records one absolute target and returns whether it held.
///
/// Wrapped so every gate in this section reports the same way and so the
/// unit is stated once beside the number rather than implied by the metric
/// name
fn gate(name: &str, value: f64, target: f64, higher_is_better: bool) -> bool {
    check_performance("lake_targets", name, value, target, higher_is_better)
}

/// Two columns of `n` rows, the shape every gate below writes
fn rows_batch(start: i64, n: usize) -> Vec<ColumnData> {
    let ids: Vec<Option<Vec<u8>>> = (0..n)
        .map(|i| Some((start + i as i64).to_le_bytes().to_vec()))
        .collect();
    let payload: Vec<Option<Vec<u8>>> = (0..n)
        .map(|i| Some(((i as i64 * 7919) % 1024).to_le_bytes().to_vec()))
        .collect();
    vec![
        ColumnData {
            column_id: 0,
            cells: ids,
        },
        ColumnData {
            column_id: 1,
            cells: payload,
        },
    ]
}

fn two_column_schema() -> LakeSchema {
    schema(&[("id", TypeId::Int64), ("bucket", TypeId::Int64)])
}

/// What one commit costs, on both of the two shapes a commit takes.
///
/// An append writes a data file and publishes a version. A predicate
/// delete writes no data at all: it records the predicate and publishes a
/// version, which is why its target is less than half the append's
#[test]
fn test_commit_latency_targets() {
    let _section = section("Target: Commit Latency");
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 900);
    let log = new_log(&paths, &two_column_schema());
    let n = commit_rows();
    tprintln!("  Rows per commit: {}", n);

    let mut append_us = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let batch = rows_batch((run * n) as i64, n);
        let (_, us) = micros(|| {
            zyron_lake::append_rows(&log, attempt(OperationKind::Append, 0), 900, &batch)
                .expect("append")
        });
        append_us.push(us);
    }
    let append_ms = record_metric("lake_targets", "Commit, insert", "us", append_us) / 1000.0;
    assert!(
        gate("Commit insert latency (ms)", append_ms, 500.0, false),
        "one commit took {:.1}ms, over the 500ms ceiling",
        append_ms
    );

    // A predicate that covers part of every file, so it is recorded rather
    // than applied and the commit is metadata only
    let mut delete_us = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let bound = (run as i64 + 1) * 3;
        let predicate = cmp(1, CompareOp::Lt, bound);
        let (_, us) = micros(|| {
            zyron_lake::delete_where(
                &log,
                attempt(OperationKind::Delete, 0),
                &predicate,
                "bucket < n",
            )
        });
        delete_us.push(us);
    }
    let delete_ms =
        record_metric("lake_targets", "Commit, delete predicate", "us", delete_us) / 1000.0;
    assert!(
        gate(
            "Commit delete predicate latency (ms)",
            delete_ms,
            200.0,
            false
        ),
        "recording a delete predicate took {:.1}ms, over the 200ms ceiling",
        delete_ms
    );
}

/// The three commits that move only metadata: adding a column, changing
/// the cluster keys, and cloning a table.
///
/// All three are the same claim in different clothes. None of them reads
/// or writes a data file, so their cost is a manifest write and nothing
/// else, and a number far above these targets means something is touching
/// data it does not need to
#[test]
fn test_metadata_only_commit_targets() {
    let _section = section("Target: Metadata-Only Commits");
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 901);
    let log = new_log(&paths, &two_column_schema());
    for run in 0..8 {
        zyron_lake::append_rows(
            &log,
            attempt(OperationKind::Append, 0),
            901,
            &rows_batch(run * 1000, 1000),
        )
        .expect("append");
    }

    // Add column: the schema grows, the files do not move
    let mut add_us = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let (_, us) = micros(|| {
            log.commit(attempt(OperationKind::SchemaChange, 0), |base| {
                let mut columns = base.schema.columns.clone();
                let id = base.schema.next_column_id;
                columns.push(LakeColumn {
                    id,
                    name: format!("added_{run}"),
                    type_id: TypeId::Int64,
                    nullable: true,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                });
                Ok(vec![LogEntry::SchemaChange(LakeSchema {
                    schema_id: base.schema.schema_id + 1,
                    next_column_id: id + 1,
                    columns,
                    derived: base.schema.derived.clone(),
                })])
            })
            .expect("add column")
        });
        add_us.push(us);
    }
    let add_ms = record_metric("lake_targets", "Schema add column", "us", add_us) / 1000.0;
    assert!(
        gate("Schema add column latency (ms)", add_ms, 500.0, false),
        "adding a column took {:.1}ms, over the 500ms ceiling",
        add_ms
    );

    // Key change: the layout a later pass will apply, with no rewrite now
    let files_before = log.latest_manifest().expect("manifest").entries.len();
    let mut alter_us = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let (_, us) = micros(|| {
            log.commit(attempt(OperationKind::SetProperty, 0), |base| {
                Ok(vec![LogEntry::SetClusterSpec(ClusterSpec {
                    spec_id: base.cluster_spec.spec_id + 1,
                    keys: vec![ClusterKey {
                        column_id: (run % 2) as u32,
                        strategy: ClusterStrategy::RangePartition,
                        param: 0,
                    }],
                })])
            })
            .expect("set cluster spec")
        });
        alter_us.push(us);
    }
    let alter_ms = record_metric("lake_targets", "Key change, no rewrite", "us", alter_us) / 1000.0;
    assert!(
        gate(
            "Key change without rewrite latency (ms)",
            alter_ms,
            50.0,
            false
        ),
        "changing the cluster keys took {:.1}ms, over the 50ms ceiling",
        alter_ms
    );
    assert_eq!(
        log.latest_manifest().expect("manifest").entries.len(),
        files_before,
        "a key change must not rewrite a file, or it is not a metadata commit"
    );

    // Clone: a whole table, in the time it takes to walk its manifest
    let mut clone_us = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let clone_paths = LakePaths::new(dir.path(), 910 + run as u32);
        let (outcome, us) = micros(|| {
            zyron_lake::clone_table(
                &log,
                clone_paths,
                attempt(OperationKind::SchemaChange, 0),
                None,
            )
            .expect("clone")
        });
        assert_eq!(
            outcome.1.files, files_before,
            "the clone has to take the whole file set"
        );
        clone_us.push(us);
    }
    let clone_ms = record_metric("lake_targets", "Clone", "us", clone_us) / 1000.0;
    assert!(
        gate("Clone latency (ms)", clone_ms, 1000.0, false),
        "cloning took {:.1}ms, over the 1s ceiling",
        clone_ms
    );
}

/// What the two rewriting passes cost: compaction over a thousand files,
/// and a clustering pass over ten million rows.
///
/// These are the two operations that read and write the whole table, so
/// they are the two whose targets are stated in tens of seconds rather
/// than milliseconds
#[test]
fn test_rewrite_pass_targets() {
    let _section = section("Target: Rewrite Passes");
    let dir = tempfile::tempdir().expect("tempdir");

    // OPTIMIZE over many small files
    let paths = LakePaths::new(dir.path(), 902);
    let log = new_log(&paths, &two_column_schema());
    let files = optimize_files();
    tprintln!("  OPTIMIZE input files: {}", files);
    for f in 0..files {
        zyron_lake::append_rows(
            &log,
            attempt(OperationKind::Append, 0),
            902,
            &rows_batch(f as i64 * 100, 100),
        )
        .expect("append");
    }
    assert_eq!(
        log.latest_manifest().expect("manifest").entries.len(),
        files
    );
    let (outcome, optimize_us) = micros(|| {
        zyron_lake::optimize(
            &log,
            attempt(OperationKind::Optimize, 0),
            902,
            zyron_lake::DEFAULT_ROWS_PER_FILE,
        )
        .expect("optimize")
    });
    assert!(
        outcome.files_removed >= files,
        "the pass has to have taken every small file, took {}",
        outcome.files_removed
    );
    record_metric(
        "lake_targets",
        "OPTIMIZE, 1000 files",
        "us",
        vec![optimize_us],
    );
    let optimize_s = optimize_us / 1_000_000.0;
    assert!(
        gate("OPTIMIZE latency (s)", optimize_s, 60.0, false),
        "compacting {} files took {:.1}s, over the 60s ceiling",
        files,
        optimize_s
    );

    // One clustering pass over a large table
    let paths = LakePaths::new(dir.path(), 903);
    let spec = ClusterSpec {
        spec_id: 1,
        keys: vec![ClusterKey {
            column_id: 1,
            strategy: ClusterStrategy::RangePartition,
            param: 0,
        }],
    };
    // Force, so what is timed is the pass rather than measurement deciding
    // whether to have one. A table with no workload behind it proposes no
    // keys, which is correct and is not what this gate is about
    let mut properties = BTreeMap::new();
    properties.insert(
        zyron_lake::CLUSTERING_MODE_PROPERTY.to_string(),
        "force".to_string(),
    );
    let log = TransactionLog::create(
        paths,
        attempt(OperationKind::SchemaChange, 0),
        &two_column_schema(),
        Some(&spec),
        &properties,
    )
    .expect("create log");
    let total = cluster_pass_rows();
    let per_file = (total / 16).max(1);
    tprintln!(
        "  Cluster pass rows: {}, files: {}",
        total,
        total / per_file
    );
    let mut written = 0usize;
    while written < total {
        let n = per_file.min(total - written);
        zyron_lake::append_rows(
            &log,
            attempt(OperationKind::Append, 0),
            903,
            &rows_batch(written as i64, n),
        )
        .expect("append");
        written += n;
    }

    let (report, pass_us) = micros(|| {
        zyron_lake::run_table_cluster_pass(
            &log,
            attempt(OperationKind::Optimize, 0),
            903,
            &zyron_lake::TablePassOptions {
                // Every file in one pass, which is what the target names
                max_inputs: 1024,
                ..zyron_lake::TablePassOptions::new(9_030_001)
            },
        )
        .expect("cluster pass")
    });
    assert!(
        report.outcome.is_some(),
        "the pass has to have run for its cost to mean anything"
    );
    record_metric(
        "lake_targets",
        "Cluster pass, 10M rows",
        "us",
        vec![pass_us],
    );
    let pass_s = pass_us / 1_000_000.0;
    assert!(
        gate("Cluster pass latency (s)", pass_s, 90.0, false),
        "a clustering pass over {} rows took {:.1}s, over the 90s ceiling",
        total,
        pass_s
    );
}

/// What the adaptive machinery costs when it is not rewriting anything:
/// deciding what to propose, and the two counters it decides from.
///
/// The two counters are the only part of clustering that touches a query's
/// hot path, so their targets are in nanoseconds. Everything else about
/// adaptive clustering happens on a maintenance thread, and a query that
/// paid measurably for being observed would make the whole feature a cost
/// rather than a saving
#[test]
fn test_adaptive_machinery_targets() {
    let _section = section("Target: Adaptive Clustering Overhead");
    let manifest = synthetic_manifest(files());
    let observer = zyron_lake::observer();
    let now = zyron_lake::current_epoch();

    // Proposal evaluation: read the evidence, choose a layout
    let mut propose_us = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (_, us) = micros(|| {
            let evidence = zyron_lake::evidence_from_manifest(&manifest, observer, 904, now);
            zyron_lake::propose(&evidence, &[], zyron_lake::DEFAULT_MAX_PROPOSED_KEYS)
        });
        propose_us.push(us);
    }
    let propose_ms = record_metric(
        "lake_targets",
        "Cluster proposal evaluation",
        "us",
        propose_us,
    ) / 1000.0;
    assert!(
        gate(
            "Cluster proposal evaluation latency (ms)",
            propose_ms,
            200.0,
            false
        ),
        "choosing a layout took {:.1}ms, over the 200ms ceiling",
        propose_ms
    );

    // The planner-side counter, per call
    let predicate = cmp(0, CompareOp::Eq, 42);
    let iterations = if measuring() { 2_000_000 } else { 20_000 };
    let (_, observe_us) = micros(|| {
        for i in 0..iterations {
            zyron_lake::observe_scan(905, &predicate, 1 << 20, (i % 1024) as u64, now);
        }
    });
    let observe_ns = observe_us * 1000.0 / iterations as f64;
    record_metric(
        "lake_targets",
        "Workload observer record",
        "ns",
        vec![observe_ns],
    );
    assert!(
        gate("Workload observer record (ns)", observe_ns, 20.0, false),
        "observing one planned scan cost {:.1}ns, over the 20ns ceiling",
        observe_ns
    );

    // The scan-completion counter, per call
    let (_, feedback_us) = micros(|| {
        for i in 0..iterations {
            zyron_lake::observe_scan_result(906, &predicate, 1024, (i % 1024) as u64, now);
        }
    });
    let feedback_ns = feedback_us * 1000.0 / iterations as f64;
    record_metric(
        "lake_targets",
        "Feedback skip-rate measurement",
        "ns",
        vec![feedback_ns],
    );
    assert!(
        gate(
            "Feedback skip-rate measurement (ns)",
            feedback_ns,
            100.0,
            false
        ),
        "measuring one finished scan cost {:.1}ns, over the 100ns ceiling",
        feedback_ns
    );
}

/// Ceiling on ordering one row by its cluster key.
///
/// Derived from six release runs, which measured 55.6, 69.8, 71.9, 78.9,
/// 79.3 and 119.7 nanoseconds a row on this machine, a mean of about
/// seventy-nine. Twice that covers every one of those observations with
/// room for a noisy machine and still trips on a doubling.
///
/// This replaced a fifty microsecond per-batch target that was written
/// against an affinity-bin placement lookup, which is constant time per
/// row. Sorting a batch is not, and no constant-time ceiling was ever
/// going to apply to it: item 3 chose to sort rather than to bin, so the
/// gate names sorting
const SORT_NS_PER_ROW_CEILING: f64 = 160.0;

/// What ordering a write by its cluster key costs, per row.
///
/// The measurement is the same subtraction it always was, appending an
/// identical batch to a table with a layout and to one without. Only the
/// unit changed: per row rather than per batch, because the work is per
/// row and a per-batch number means nothing without the batch size beside
/// it
#[test]
fn test_cluster_key_sort_overhead_target() {
    let _section = section("Target: Cluster Key Sort Overhead");
    let dir = tempfile::tempdir().expect("tempdir");
    let n = commit_rows();

    let plain = new_log(&LakePaths::new(dir.path(), 907), &two_column_schema());
    let spec = ClusterSpec {
        spec_id: 1,
        keys: vec![ClusterKey {
            column_id: 1,
            strategy: ClusterStrategy::RangePartition,
            param: 0,
        }],
    };
    let clustered = TransactionLog::create(
        LakePaths::new(dir.path(), 908),
        attempt(OperationKind::SchemaChange, 0),
        &two_column_schema(),
        Some(&spec),
        &BTreeMap::new(),
    )
    .expect("create log");

    let mut plain_us = 0f64;
    let mut clustered_us = 0f64;
    for run in 0..RUNS {
        let batch = rows_batch((run * n) as i64, n);
        // Alternated, so cache and thermal state do not favour whichever
        // side runs first
        if run % 2 == 0 {
            plain_us += micros(|| {
                zyron_lake::append_rows(&plain, attempt(OperationKind::Append, 0), 907, &batch)
                    .expect("append")
            })
            .1;
            clustered_us += micros(|| {
                zyron_lake::append_rows(&clustered, attempt(OperationKind::Append, 0), 908, &batch)
                    .expect("append")
            })
            .1;
        } else {
            clustered_us += micros(|| {
                zyron_lake::append_rows(&clustered, attempt(OperationKind::Append, 0), 908, &batch)
                    .expect("append")
            })
            .1;
            plain_us += micros(|| {
                zyron_lake::append_rows(&plain, attempt(OperationKind::Append, 0), 907, &batch)
                    .expect("append")
            })
            .1;
        }
    }
    let per_row_ns = ((clustered_us - plain_us) / RUNS as f64).max(0.0) * 1000.0 / n as f64;
    tprintln!("  Rows per append: {}", n);
    record_metric(
        "lake_targets",
        "Cluster key sort overhead",
        "ns/row",
        vec![per_row_ns],
    );
    assert!(
        gate(
            "Cluster key sort overhead (ns per row)",
            per_row_ns,
            SORT_NS_PER_ROW_CEILING,
            false
        ),
        "ordering a row by its cluster key cost {:.1}ns",
        per_row_ns
    );

    // Where that cost goes, recorded rather than gated: it explains the
    // number above and is not a second claim. Alternated and averaged,
    // because a single unpaired sample of each reported anything from zero
    // to sixty nanoseconds a row across runs
    let ordered_batch: Vec<ColumnData> = {
        let ids: Vec<Option<Vec<u8>>> = (0..n)
            .map(|i| Some((1_000_000i64 + i as i64).to_le_bytes().to_vec()))
            .collect();
        let bucket: Vec<Option<Vec<u8>>> = (0..n)
            .map(|i| Some((i as i64).to_le_bytes().to_vec()))
            .collect();
        vec![
            ColumnData {
                column_id: 0,
                cells: ids,
            },
            ColumnData {
                column_id: 1,
                cells: bucket,
            },
        ]
    };
    let mut presorted_total = 0f64;
    let mut shuffled_total = 0f64;
    for run in 0..RUNS {
        let shuffled = rows_batch((500_000 + run * n) as i64, n);
        let mut once = |batch: &[ColumnData]| {
            micros(|| {
                zyron_lake::append_rows(&clustered, attempt(OperationKind::Append, 0), 908, batch)
                    .expect("append")
            })
            .1
        };
        if run % 2 == 0 {
            presorted_total += once(&ordered_batch);
            shuffled_total += once(&shuffled);
        } else {
            shuffled_total += once(&shuffled);
            presorted_total += once(&ordered_batch);
        }
    }
    record_metric(
        "lake_targets",
        "Clustered append, batch already in key order",
        "us",
        vec![presorted_total / RUNS as f64],
    );
    record_metric(
        "lake_targets",
        "Clustered append, batch shuffled",
        "us",
        vec![shuffled_total / RUNS as f64],
    );
}

/// How long it takes to notice a table needs repairing ahead of its clock.
///
/// The fast lane is what item 3 added: a table whose drift has crossed its
/// urgency threshold is passed on the next tick whatever its interval says.
/// What is timed is the decision, which is the part this code owns. The
/// other component of "how soon does repair start" is the tick interval,
/// which is configuration rather than code, and gating it would be gating
/// a number an operator chose
#[test]
fn test_fast_lane_trigger_latency_target() {
    let _section = section("Target: Repair Fast Lane Trigger");
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 913);
    let spec = ClusterSpec {
        spec_id: 1,
        keys: vec![ClusterKey {
            column_id: 1,
            strategy: ClusterStrategy::RangePartition,
            param: 0,
        }],
    };
    let mut properties = BTreeMap::new();
    properties.insert(
        zyron_lake::CLUSTER_REPAIR_URGENCY_THRESHOLD_PROPERTY.to_string(),
        "8".to_string(),
    );
    let log = TransactionLog::create(
        paths,
        attempt(OperationKind::SchemaChange, 0),
        &two_column_schema(),
        Some(&spec),
        &properties,
    )
    .expect("create log");

    // Overlapping files: every append spans the same key range, which is
    // exactly the drift the threshold counts
    let overlapping = 12;
    for run in 0..overlapping {
        let batch = rows_batch(run as i64 * 10, 300);
        zyron_lake::append_rows(&log, attempt(OperationKind::Append, 0), 913, &batch)
            .expect("append");
    }
    let manifest = log.latest_manifest().expect("manifest");
    let drifted = zyron_lake::drifted_file_count(&manifest);
    assert!(
        drifted > manifest.cluster_repair_urgency_threshold(),
        "the workload has to actually cross the threshold: {} drifted against a threshold of {}",
        drifted,
        manifest.cluster_repair_urgency_threshold()
    );

    // What a maintenance tick pays to find that out, per table
    let mut decide_us = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (fires, us) = micros(|| {
            let cold =
                TransactionLog::open(LakePaths::new(dir.path(), 913), &AllCommitted).expect("open");
            let manifest = cold.latest_manifest().expect("manifest");
            zyron_lake::drifted_file_count(&manifest) > manifest.cluster_repair_urgency_threshold()
        });
        assert!(fires, "a table over its threshold has to be picked up");
        decide_us.push(us);
    }
    let decide_ms = record_metric(
        "lake_targets",
        "Fast lane trigger decision",
        "us",
        decide_us,
    ) / 1000.0;
    assert!(
        gate("Fast lane trigger latency (ms)", decide_ms, 100.0, false),
        "deciding that a table needs repairing took {:.1}ms",
        decide_ms
    );
}

/// That `cluster_repair_max_inputs` does what it says.
///
/// A tunable nobody can observe is a tunable nobody should trust. Two
/// identical tables, two settings, and the pass with four times the bound
/// has to take close to four times the files. Gated at three and a half
/// rather than four, because the bound is a ceiling on what a pass may
/// take and not a promise about what it will find
#[test]
fn test_repair_max_inputs_scaling_target() {
    let _section = section("Target: Repair Max Inputs Scaling");
    let dir = tempfile::tempdir().expect("tempdir");

    let build = |table_id: u32| -> TransactionLog {
        let spec = ClusterSpec {
            spec_id: 1,
            keys: vec![ClusterKey {
                column_id: 1,
                strategy: ClusterStrategy::RangePartition,
                param: 0,
            }],
        };
        let mut properties = BTreeMap::new();
        properties.insert(
            zyron_lake::CLUSTERING_MODE_PROPERTY.to_string(),
            "force".to_string(),
        );
        let log = TransactionLog::create(
            LakePaths::new(dir.path(), table_id),
            attempt(OperationKind::SchemaChange, 0),
            &two_column_schema(),
            Some(&spec),
            &properties,
        )
        .expect("create log");
        // Enough overlapping files that the larger bound is the thing that
        // limits the pass, not the supply
        for run in 0..80 {
            let batch = rows_batch(run as i64 * 10, 200);
            zyron_lake::append_rows(
                &log,
                attempt(OperationKind::Append, 0),
                table_id as u64,
                &batch,
            )
            .expect("append");
        }
        log
    };

    let run_with =
        |log: &TransactionLog, table_id: u32, max_inputs: usize, pass_id: u64| -> usize {
            let report = zyron_lake::run_table_cluster_pass(
                log,
                attempt(OperationKind::Optimize, 0),
                table_id,
                &zyron_lake::TablePassOptions {
                    max_inputs,
                    ..zyron_lake::TablePassOptions::new(pass_id)
                },
            )
            .expect("cluster pass");
            report.outcome.map(|o| o.inputs).unwrap_or(0)
        };

    let narrow_log = build(914);
    let wide_log = build(915);
    let narrow = run_with(&narrow_log, 914, 16, 9_140_001);
    let wide = run_with(&wide_log, 915, 64, 9_150_001);
    tprintln!(
        "  Files taken: {} at max_inputs 16, {} at max_inputs 64",
        narrow,
        wide
    );
    assert!(
        narrow > 0 && wide > 0,
        "both passes have to have run for the ratio to mean anything"
    );
    let ratio = wide as f64 / narrow as f64;
    record_metric(
        "lake_targets",
        "Repair max_inputs scaling",
        "x",
        vec![ratio],
    );
    assert!(
        gate("Repair max_inputs scaling (x)", ratio, 3.5, true),
        "four times the bound took {:.2}x the files, so the tunable is not doing what it says",
        ratio
    );
}

/// What reaching back through history costs, absolutely.
///
/// Six attempts to state this as a percentage, and the sixth is why it is
/// not one. Recorded so none of them is tried again:
///
/// 1. `manifest_at(old)` against `manifest_at(head)` on one log compared
///    two cache hits and reported zero.
/// 2. The same on cold logs compared a real reconstruction against a head
///    that opening the log had already materialized, and reported tens of
///    thousands of percent.
/// 3. Comparing whole reads of two versions compared different amounts of
///    data, because an older version names fewer files.
/// 4. Measuring one version twice, once as head and once buried, measured
///    `TransactionLog::open` scanning a longer log directory.
/// 5. Comparing both on one log removed that, but divided a fixed
///    resolution cost by a whole read of four thousand rows, which is the
///    smallest thing anyone would ever time travel for. Any fixed cost
///    over a denominator that small reads as a large percentage.
/// 6. There is no denominator. A percentage needs a representative query
///    and this suite has no way to say what one is, so the cost is stated
///    as what it is: microseconds to reach back, and microseconds per
///    version reached back.
///
/// The second of those is the one that would catch a real regression. A
/// linear replay stays flat per version however far back it goes, and a
/// hidden non-linear term shows up there long before it shows up in the
/// total
#[test]
fn test_time_travel_reach_back_targets() {
    let _section = section("Target: Time Travel Reach Back");
    let dir = tempfile::tempdir().expect("tempdir");
    let n = commit_rows();
    let paths = LakePaths::new(dir.path(), 912);
    let log = new_log(&paths, &two_column_schema());
    for run in 0..4 {
        zyron_lake::append_rows(
            &log,
            attempt(OperationKind::Append, 0),
            912,
            &rows_batch((run * n) as i64, n),
        )
        .expect("append");
    }
    let pinned = log.head_version();

    let buried_versions: u64 = if measuring() { 200 } else { 20 };
    for run in 0..buried_versions {
        zyron_lake::append_rows(
            &log,
            attempt(OperationKind::Append, 0),
            912,
            &rows_batch((1_000_000 + run * 100) as i64, 100),
        )
        .expect("append");
    }
    // The shape a running server keeps a log in. Without it every resolve
    // replays from the beginning of time, which is a number about the
    // absence of maintenance rather than about the format
    checkpoint_like_the_server(&log);

    let resolve = |version: u64| -> f64 {
        micros(|| {
            let cold = TransactionLog::open(paths.clone(), &AllCommitted).expect("open");
            cold.manifest_at(version).expect("manifest")
        })
        .1
    };
    let head = log.head_version();
    let mut reached = Vec::with_capacity(RUNS);
    let mut newest = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        // Both on the same log, so each pays the same open cost and the
        // subtraction removes it. Alternated so neither side runs on a
        // systematically warmer page cache
        if run % 2 == 0 {
            reached.push(resolve(pinned));
            newest.push(resolve(head));
        } else {
            newest.push(resolve(head));
            reached.push(resolve(pinned));
        }
    }
    let reached_us = reached.iter().sum::<f64>() / RUNS as f64;
    let newest_us = newest.iter().sum::<f64>() / RUNS as f64;
    tprintln!(
        "  Version {} resolved in {:.1}us, head {} in {:.1}us, over {} versions of history",
        pinned,
        reached_us,
        head,
        newest_us,
        buried_versions
    );

    // The older version names far fewer files than the head does, so
    // content cost biases this downward. A positive number is therefore a
    // lower bound on what reaching back costs
    let delta_us = (reached_us - newest_us).max(0.0);
    record_metric(
        "lake_targets",
        "Time travel reach back delta",
        "us",
        vec![delta_us],
    );
    assert!(
        gate("Time travel reach back delta (us)", delta_us, 1500.0, false),
        "reaching back {} versions cost {:.1}us",
        buried_versions,
        delta_us
    );

    let per_version_us = delta_us / buried_versions as f64;
    record_metric(
        "lake_targets",
        "Time travel reach back per version",
        "us/version",
        vec![per_version_us],
    );
    assert!(
        gate(
            "Time travel reach back per version (us)",
            per_version_us,
            10.0,
            false
        ),
        "reaching back cost {:.2}us a version, which is where a non-linear replay would show",
        per_version_us
    );
}

/// What it costs to open a table with a lot of files, and how many commits
/// a second the log will take.
///
/// The first is what every reader pays before it reads anything. The
/// second is what bounds a write workload, and it is a throughput rather
/// than a latency because optimistic concurrency trades one for the other
#[test]
fn test_manifest_and_concurrency_targets() {
    let _section = section("Target: Manifest Load and Commit Throughput");
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = LakePaths::new(dir.path(), 909);
    let log = new_log(&paths, &two_column_schema());
    let files = manifest_files();
    tprintln!("  Files in the manifest: {}", files);

    // Built through real commits, so what is reconstructed is a real log
    // rather than a manifest somebody assembled
    for _ in 0..files {
        log.commit(attempt(OperationKind::Append, 0), |base| {
            let mut entry = synthetic_manifest(1).entries[0].clone();
            entry.partition_id = base.entries.len() as u64 + 1;
            Ok(vec![LogEntry::AddFile(entry)])
        })
        .expect("commit");
    }
    // The shape a running server keeps this table in. Without it the ten
    // thousand commits are ten thousand entries to replay, and the number
    // measures a log that has never been maintained rather than a manifest
    // that names ten thousand files
    checkpoint_like_the_server(&log);
    let head = log.head_version();
    let mut load_us = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        // The open is inside the timing, because opening the log is what
        // reconstructs the manifest. Timing only the call after it measures
        // a cache lookup, which is not what a reader pays to open a table
        let (manifest, us) = micros(|| {
            let cold = TransactionLog::open(paths.clone(), &AllCommitted).expect("open");
            cold.manifest_at(head).expect("manifest")
        });
        assert_eq!(manifest.entries.len(), files);
        load_us.push(us);
    }
    let load_ms = record_metric("lake_targets", "Manifest load, 10K files", "us", load_us) / 1000.0;
    assert!(
        gate("Manifest load latency (ms)", load_ms, 200.0, false),
        "opening a table with {} files took {:.1}ms, over the 200ms ceiling",
        files,
        load_ms
    );

    // Commits per second, serialized through exclusive file creation
    let commits = if measuring() { 200 } else { 20 };
    let paths = LakePaths::new(dir.path(), 911);
    let log = new_log(&paths, &two_column_schema());
    let (_, total_us) = micros(|| {
        for i in 0..commits {
            zyron_lake::append_rows(
                &log,
                attempt(OperationKind::Append, 0),
                911,
                &rows_batch(i as i64 * 10, 10),
            )
            .expect("append");
        }
    });
    let per_sec = commits as f64 / (total_us / 1_000_000.0);
    record_metric(
        "lake_targets",
        "Concurrent commits",
        " commits/sec",
        vec![per_sec],
    );
    assert!(
        gate("Commit throughput (commits/sec)", per_sec, 10.0, true),
        "the log took {:.1} commits a second, under the 10/sec floor",
        per_sec
    );
}

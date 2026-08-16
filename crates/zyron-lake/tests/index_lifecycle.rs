//! Secondary index behaviour across the operations that move rows.
//!
//! The load-bearing property is that an index can be behind but never
//! wrong. Every test here either proves a probe returns exactly the rows a
//! scan would, or proves the index declines rather than answering short.

use std::collections::BTreeMap;

use zyron_common::TypeId;
use zyron_lake::operations::{create_index, drop_index, rebuild_indexes};
use zyron_lake::schema::LakeColumn;
use zyron_lake::{
    ColumnData, CommitAttempt, CompareOp, LakeFileReader, LakePaths, LakePredicate, LakeSchema,
    LakeValue, OperationKind, TransactionLog, append_rows, covers_table, delete_where,
    group_by_partition, optimize, probe_equal, update_where, vacuum_data_files,
};

const TABLE_ID: u32 = 42;

fn schema() -> LakeSchema {
    LakeSchema::new(
        1,
        vec![
            LakeColumn {
                id: 0,
                name: "id".into(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            },
            LakeColumn {
                id: 1,
                name: "email".into(),
                type_id: TypeId::Varchar,
                nullable: true,
                fractional_digits: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            },
        ],
    )
    .expect("schema")
}

fn attempt(timestamp_us: i64) -> CommitAttempt<'static> {
    CommitAttempt {
        operation: OperationKind::Append,
        db_txn_id: 0,
        commit_lsn: 1,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
    }
}

fn new_log(dir: &std::path::Path) -> TransactionLog {
    TransactionLog::create(
        LakePaths::new(dir, TABLE_ID),
        CommitAttempt {
            operation: OperationKind::SchemaChange,
            ..attempt(100)
        },
        &schema(),
        None,
        &BTreeMap::new(),
    )
    .expect("create")
}

/// Rows with `id = n` and `email = "user<n>@example.com"`
fn rows(ids: &[i64]) -> Vec<ColumnData> {
    vec![
        ColumnData {
            column_id: 0,
            cells: ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
        },
        ColumnData {
            column_id: 1,
            cells: ids
                .iter()
                .map(|v| Some(format!("user{}@example.com", v).into_bytes()))
                .collect(),
        },
    ]
}

fn email_of(id: i64) -> Vec<u8> {
    format!("user{}@example.com", id).into_bytes()
}

/// Every live row of the table, read the way a scan reads it, as
/// (id, email) pairs sorted by id
fn scan_all(log: &TransactionLog) -> Vec<(i64, String)> {
    let manifest = log.latest_manifest().expect("manifest");
    let mut out = Vec::new();
    for entry in &manifest.entries {
        let reader = LakeFileReader::open(log.paths(), entry.partition_id).expect("open");
        let keep = reader
            .delete_survivors(&manifest.schema, &manifest, entry)
            .expect("survivors");
        let ids = reader.read_column(&manifest.schema.columns[0]).expect("id");
        let emails = reader
            .read_column(&manifest.schema.columns[1])
            .expect("email");
        for row in 0..reader.row_count() {
            if keep[row / 8] & (1 << (row % 8)) == 0 {
                continue;
            }
            let mut a = [0u8; 8];
            a.copy_from_slice(ids.cell(row).expect("id cell"));
            out.push((
                i64::from_le_bytes(a),
                String::from_utf8_lossy(emails.cell(row).unwrap_or(b"")).into_owned(),
            ));
        }
    }
    out.sort();
    out
}

/// The rows an equality probe on the email index resolves to, read back
/// through the data files exactly as a fetch would
fn probe_rows(log: &TransactionLog, email: &[u8]) -> Vec<(i64, String)> {
    let manifest = log.latest_manifest().expect("manifest");
    let spec = manifest.index_by_name("ix_email").expect("index declared");
    assert!(
        covers_table(&manifest, spec.index_id),
        "index does not cover the table, a probe would answer short"
    );
    let (addresses, _) = probe_equal(log.paths(), &manifest, spec, &[Some(email)]).expect("probe");

    let mut out = Vec::new();
    for (partition_id, ordinals) in group_by_partition(&addresses) {
        let entry = manifest
            .entry_for(partition_id)
            .expect("probe named a live file");
        let reader = LakeFileReader::open(log.paths(), partition_id).expect("open");
        let keep = reader
            .delete_survivors(&manifest.schema, &manifest, entry)
            .expect("survivors");
        let ids = reader.read_column(&manifest.schema.columns[0]).expect("id");
        let emails = reader
            .read_column(&manifest.schema.columns[1])
            .expect("email");
        for ordinal in ordinals {
            let row = ordinal as usize;
            // A probe may address a row a later delete predicate removed,
            // so the fetch filters exactly as a scan does
            if keep[row / 8] & (1 << (row % 8)) == 0 {
                continue;
            }
            let mut a = [0u8; 8];
            a.copy_from_slice(ids.cell(row).expect("id cell"));
            out.push((
                i64::from_le_bytes(a),
                String::from_utf8_lossy(emails.cell(row).unwrap_or(b"")).into_owned(),
            ));
        }
    }
    out.sort();
    out
}

fn create_email_index(log: &TransactionLog) {
    create_index(log, attempt(300), TABLE_ID as u64, "ix_email", &[1], false)
        .expect("create index");
}

#[test]
fn test_a_probe_returns_exactly_what_a_scan_would() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    for (n, chunk) in [vec![1i64, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        .into_iter()
        .enumerate()
    {
        append_rows(
            &log,
            attempt(200 + n as i64),
            TABLE_ID as u64,
            &rows(&chunk),
        )
        .expect("append");
    }
    create_email_index(&log);

    // Every stored key resolves to its own row and nothing else
    for id in 1..=9i64 {
        assert_eq!(
            probe_rows(&log, &email_of(id)),
            vec![(id, format!("user{}@example.com", id))],
            "probing user{}",
            id
        );
    }
    // A key the table does not hold resolves to nothing
    assert!(probe_rows(&log, b"absent@example.com").is_empty());
}

#[test]
fn test_an_append_after_the_index_exists_is_indexed_by_the_same_commit() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2])).expect("append");
    create_email_index(&log);

    let version_before = log.latest_version();
    append_rows(&log, attempt(400), TABLE_ID as u64, &rows(&[7, 8])).expect("append");
    assert_eq!(
        log.latest_version(),
        version_before + 1,
        "the rows and their index entries are one commit, never two"
    );

    // Coverage is complete the instant the rows are visible, so no version
    // exists where a probe would answer short
    let manifest = log.latest_manifest().expect("manifest");
    let spec = manifest.index_by_name("ix_email").expect("index");
    assert!(covers_table(&manifest, spec.index_id));
    assert_eq!(
        probe_rows(&log, &email_of(7)),
        vec![(7, "user7@example.com".to_string())]
    );
    assert_eq!(
        probe_rows(&log, &email_of(1)),
        vec![(1, "user1@example.com".to_string())],
        "the pre-existing rows still resolve"
    );
}

#[test]
fn test_a_probe_never_returns_a_row_a_delete_predicate_removed() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2, 3])).expect("append");
    create_email_index(&log);

    delete_where(
        &log,
        CommitAttempt {
            operation: OperationKind::Delete,
            ..attempt(500)
        },
        &LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Eq,
            value: LakeValue::Int(2),
        },
        "id = 2",
    )
    .expect("delete");

    // The index still holds an entry for the deleted row, and the fetch
    // filters it out exactly as a scan does. Probe and scan agree
    assert!(probe_rows(&log, &email_of(2)).is_empty());
    assert_eq!(
        probe_rows(&log, &email_of(3)),
        vec![(3, "user3@example.com".to_string())]
    );
    assert_eq!(scan_all(&log).len(), 2);
}

#[test]
fn test_an_update_reindexes_the_rows_it_rewrote() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2, 3])).expect("append");
    create_email_index(&log);

    // Replace row 2's email. The old file goes, a new one arrives
    let predicate = LakePredicate::Compare {
        column_id: 0,
        op: CompareOp::Eq,
        value: LakeValue::Int(2),
    };
    let replacement = vec![
        ColumnData {
            column_id: 0,
            cells: vec![Some(2i64.to_le_bytes().to_vec())],
        },
        ColumnData {
            column_id: 1,
            cells: vec![Some(b"changed@example.com".to_vec())],
        },
    ];
    update_where(
        &log,
        CommitAttempt {
            operation: OperationKind::Update,
            ..attempt(600)
        },
        TABLE_ID as u64,
        Some(&predicate),
        "id = 2",
        &replacement,
        1,
    )
    .expect("update");

    assert_eq!(
        probe_rows(&log, b"changed@example.com"),
        vec![(2, "changed@example.com".to_string())],
        "the new image is indexed"
    );
    assert!(
        probe_rows(&log, &email_of(2)).is_empty(),
        "the replaced image is gone"
    );
}

#[test]
fn test_optimize_keeps_the_index_complete_across_the_rewrite() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2, 3, 4, 5])).expect("append");
    create_email_index(&log);
    delete_where(
        &log,
        CommitAttempt {
            operation: OperationKind::Delete,
            ..attempt(500)
        },
        &LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Lt,
            value: LakeValue::Int(3),
        },
        "id < 3",
    )
    .expect("delete");

    let outcome = optimize(
        &log,
        CommitAttempt {
            operation: OperationKind::Optimize,
            ..attempt(700)
        },
        TABLE_ID as u64,
    )
    .expect("optimize");
    assert!(outcome.version.is_some(), "the rewrite happened");

    // Every surviving row still resolves through the index, addressed in
    // the file the rewrite produced rather than the one it removed
    let manifest = log.latest_manifest().expect("manifest");
    let spec = manifest.index_by_name("ix_email").expect("index");
    assert!(
        covers_table(&manifest, spec.index_id),
        "a rewrite that left the index behind would decline here"
    );
    for id in 3..=5i64 {
        assert_eq!(
            probe_rows(&log, &email_of(id)),
            vec![(id, format!("user{}@example.com", id))]
        );
    }
    assert!(probe_rows(&log, &email_of(1)).is_empty());
}

#[test]
fn test_a_past_version_reads_the_index_that_version_had() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2])).expect("append");
    create_email_index(&log);
    let at_two_rows = log.latest_version();

    append_rows(&log, attempt(400), TABLE_ID as u64, &rows(&[3, 4])).expect("append");
    let at_four_rows = log.latest_version();

    // The older manifest names the index files that version had, so a
    // probe against it sees two rows and not four
    let old = log.manifest_at(at_two_rows).expect("old manifest");
    let spec = old.index_by_name("ix_email").expect("index");
    assert!(covers_table(&old, spec.index_id));
    let (found, _) = probe_equal(log.paths(), &old, spec, &[Some(&email_of(3))]).expect("probe");
    assert!(
        found.is_empty(),
        "row 3 did not exist at version {}",
        at_two_rows
    );
    let (found, _) = probe_equal(log.paths(), &old, spec, &[Some(&email_of(1))]).expect("probe");
    assert_eq!(found.len(), 1, "row 1 did exist");

    let new = log.manifest_at(at_four_rows).expect("new manifest");
    let spec = new.index_by_name("ix_email").expect("index");
    let (found, _) = probe_equal(log.paths(), &new, spec, &[Some(&email_of(3))]).expect("probe");
    assert_eq!(found.len(), 1, "row 3 exists at the newer version");
}

#[test]
fn test_an_index_declines_rather_than_answering_short() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2])).expect("append");
    create_email_index(&log);

    // A manifest carrying a data file no index file covers is exactly the
    // state a missed maintenance hook would produce. Coverage reports it
    // rather than letting a probe answer from the files it does have
    let mut manifest = (*log.latest_manifest().expect("manifest")).clone();
    let spec = manifest.index_by_name("ix_email").expect("index").clone();
    assert!(covers_table(&manifest, spec.index_id));

    let mut unindexed = manifest.entries[0].clone();
    unindexed.partition_id = 0xDEAD_BEEF;
    manifest.entries.push(unindexed);
    manifest.entries.sort_by_key(|e| e.partition_id);
    assert!(
        !covers_table(&manifest, spec.index_id),
        "an uncovered data file must make the index decline"
    );
}

#[test]
fn test_probe_statistics_prove_the_index_pruned_rather_than_scanned() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    // Several appends, so the index holds several files and a probe has to
    // choose between them
    for (n, chunk) in [vec![1i64, 2, 3], vec![100, 101, 102], vec![200, 201, 202]]
        .into_iter()
        .enumerate()
    {
        append_rows(
            &log,
            attempt(200 + n as i64),
            TABLE_ID as u64,
            &rows(&chunk),
        )
        .expect("append");
    }
    create_email_index(&log);
    // One more append after the build, so the index has a second file
    append_rows(&log, attempt(400), TABLE_ID as u64, &rows(&[300, 301])).expect("append");

    let manifest = log.latest_manifest().expect("manifest");
    let spec = manifest.index_by_name("ix_email").expect("index");
    let (addresses, stats) =
        probe_equal(log.paths(), &manifest, spec, &[Some(&email_of(201))]).expect("probe");
    assert_eq!(addresses.len(), 1);
    assert!(
        stats.files_considered >= 2,
        "the index holds more than one file"
    );
    assert!(
        stats.entries_examined <= 8,
        "examined {} entries for one key, a walk would have read every one",
        stats.entries_examined
    );
}

#[test]
fn test_a_large_index_splits_into_range_disjoint_files_and_a_probe_opens_one() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    // More entries than one index file holds, so the build has to split
    let total = zyron_lake::ENTRIES_PER_INDEX_FILE * 3;
    let ids: Vec<i64> = (0..total as i64).collect();
    for chunk in ids.chunks(4096) {
        append_rows(&log, attempt(200), TABLE_ID as u64, &rows(chunk)).expect("append");
    }
    create_email_index(&log);

    let manifest = log.latest_manifest().expect("manifest");
    let spec = manifest.index_by_name("ix_email").expect("index");
    let files: Vec<&zyron_lake::IndexFileEntry> = manifest
        .index_files
        .iter()
        .filter(|f| f.index_id == spec.index_id)
        .collect();
    assert!(
        files.len() >= 3,
        "an index of {} entries must span more than one file, got {}",
        total,
        files.len()
    );
    // A file may exceed the target when one sort-key run does. These keys
    // are distinct strings that still share a truncated sort key in long
    // runs, and splitting such a run is what would make the ranges overlap,
    // so the disjointness below is the property worth holding rather than
    // a hard cap on file size

    // Ranges are disjoint, which is what lets the manifest pick one file.
    // The key here is a string, so the bounds compare as strings
    let text = |v: &zyron_lake::LakeValue| match v {
        zyron_lake::LakeValue::Str(s) => s.clone(),
        other => panic!("expected a string bound, got {other:?}"),
    };
    let mut bounds: Vec<(String, String)> = files
        .iter()
        .filter_map(|f| {
            let stats = f.file.stats_for(0)?;
            Some((
                text(stats.bounds.min.as_ref()?),
                text(stats.bounds.max.as_ref()?),
            ))
        })
        .collect();
    bounds.sort();
    for pair in bounds.windows(2) {
        assert!(
            pair[0].1 < pair[1].0,
            "index file ranges overlap at {} and {}, so a probe cannot prune to one file",
            pair[0].1,
            pair[1].0
        );
    }

    // A probe opens one file, not the whole index
    let (addresses, stats) =
        probe_equal(log.paths(), &manifest, spec, &[Some(&email_of(12_345))]).expect("probe");
    assert_eq!(addresses.len(), 1);
    assert_eq!(
        stats.files_opened,
        1,
        "opened {} of {} index files for one key",
        stats.files_opened,
        files.len()
    );
    // Bisection lands on the run sharing the key's truncated sort key and
    // the exact comparison walks it, so the work is the run length rather
    // than the file length
    assert!(
        stats.entries_examined < 64,
        "examined {} entries for one key, which is a walk rather than a bisection",
        stats.entries_examined
    );
}

#[test]
fn test_a_range_probe_returns_exactly_the_rows_in_the_range() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    let ids: Vec<i64> = (0..600).collect();
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&ids)).expect("append");
    // Index the integer key so the range has a numeric order to bound
    create_index(&log, attempt(300), TABLE_ID as u64, "ix_id", &[0], false).expect("create index");

    let manifest = log.latest_manifest().expect("manifest");
    let spec = manifest.index_by_name("ix_id").expect("index");
    let bound = |v: i64, inclusive: bool| zyron_lake::index::RangeBound {
        value: LakeValue::Int(v),
        inclusive,
    };

    // A closed range returns exactly its members
    let (addresses, _) = zyron_lake::probe_range(
        log.paths(),
        &manifest,
        spec,
        Some(&bound(100, true)),
        Some(&bound(109, true)),
    )
    .expect("range probe");
    assert_eq!(addresses.len(), 10, "ten keys are in [100, 109]");

    // Exclusive ends drop their own endpoints and nothing else
    let (exclusive, _) = zyron_lake::probe_range(
        log.paths(),
        &manifest,
        spec,
        Some(&bound(100, false)),
        Some(&bound(109, false)),
    )
    .expect("range probe");
    assert_eq!(exclusive.len(), 8);

    // An open upper side runs to the end of the index
    let (open, _) =
        zyron_lake::probe_range(log.paths(), &manifest, spec, Some(&bound(590, true)), None)
            .expect("range probe");
    assert_eq!(open.len(), 10);

    // A range past every stored key selects nothing and opens no file
    let (empty, stats) = zyron_lake::probe_range(
        log.paths(),
        &manifest,
        spec,
        Some(&bound(10_000, true)),
        None,
    )
    .expect("range probe");
    assert!(empty.is_empty());
    assert_eq!(stats.files_opened, 0, "bounds alone answered");
}

#[test]
fn test_fragmented_index_runs_are_compacted_back_into_disjoint_ranges() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[0])).expect("append");
    create_email_index(&log);

    // Many small writes, each appending its own index file over an
    // overlapping key range, which is what stops the manifest pruning
    for i in 1..40i64 {
        append_rows(
            &log,
            attempt(300 + i),
            TABLE_ID as u64,
            &rows(&[i * 7 % 97]),
        )
        .expect("append");
    }
    let before = log.latest_manifest().expect("manifest").index_files.len();
    assert!(before > 8, "the runs did not accumulate, got {}", before);

    let compacted = zyron_lake::operations::compact_indexes_if_fragmented(
        &log,
        CommitAttempt {
            operation: OperationKind::SchemaChange,
            ..attempt(900)
        },
        TABLE_ID as u64,
    )
    .expect("compact");
    assert!(compacted.is_some(), "fragmentation was not recognized");

    let manifest = log.latest_manifest().expect("manifest");
    assert!(
        manifest.index_files.len() < before,
        "compaction did not reduce the run count"
    );
    let spec = manifest.index_by_name("ix_email").expect("index");
    assert!(covers_table(&manifest, spec.index_id));

    // A compact index is left alone
    assert!(
        zyron_lake::operations::compact_indexes_if_fragmented(
            &log,
            CommitAttempt {
                operation: OperationKind::SchemaChange,
                ..attempt(1000)
            },
            TABLE_ID as u64,
        )
        .expect("compact")
        .is_none(),
        "a compact index was rebuilt for no reason"
    );
}

#[test]
fn test_dropping_an_index_frees_its_files_and_stops_maintaining_it() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2, 3])).expect("append");
    create_email_index(&log);
    assert_eq!(
        log.latest_manifest().expect("manifest").index_files.len(),
        1
    );

    drop_index(
        &log,
        CommitAttempt {
            operation: OperationKind::SchemaChange,
            ..attempt(800)
        },
        "ix_email",
    )
    .expect("drop");
    let manifest = log.latest_manifest().expect("manifest");
    assert!(manifest.indexes.is_empty());
    assert!(
        manifest.index_files.is_empty(),
        "a dropped index takes its files with it"
    );

    // The rows are untouched, and vacuum reclaims the index file once no
    // retained version names it
    assert_eq!(scan_all(&log).len(), 3);
    vacuum_data_files(&log, log.latest_version()).expect("vacuum");
    assert_eq!(scan_all(&log).len(), 3, "vacuum took no data file with it");
}

#[test]
fn test_rebuild_replaces_every_index_file_and_still_answers() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(&log, attempt(200), TABLE_ID as u64, &rows(&[1, 2, 3])).expect("append");
    create_email_index(&log);
    append_rows(&log, attempt(300), TABLE_ID as u64, &rows(&[4, 5])).expect("append");
    let before: Vec<u64> = log
        .latest_manifest()
        .expect("manifest")
        .index_files
        .iter()
        .map(|f| f.file.partition_id)
        .collect();
    assert_eq!(before.len(), 2, "one file per commit before the rebuild");

    rebuild_indexes(
        &log,
        CommitAttempt {
            operation: OperationKind::SchemaChange,
            ..attempt(900)
        },
        TABLE_ID as u64,
    )
    .expect("rebuild");

    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(
        manifest.index_files.len(),
        1,
        "the rebuild collapsed the runs into one"
    );
    assert!(
        manifest
            .index_files
            .iter()
            .all(|f| !before.contains(&f.file.partition_id)),
        "every file was replaced"
    );
    for id in 1..=5i64 {
        assert_eq!(
            probe_rows(&log, &email_of(id)),
            vec![(id, format!("user{}@example.com", id))]
        );
    }
}

//! A lake table's changes, derived from its transaction log.
//!
//! What these prove is that the log is the change record. An append yields
//! its rows as inserts, a delete yields the rows its predicate matched as
//! deletes and nothing else, an update yields both sides under one version,
//! a compaction yields no change at all, and a change read goes exactly as
//! far back as time travel does, so a stream position below the oldest
//! version the log still holds is stale precisely when AS OF that version
//! would fail.
//!
//! Run: cargo test -p zyron-lake --test lake_change_feed_test -- --nocapture

use std::collections::BTreeMap;

use zyron_common::TypeId;
use zyron_lake::{
    AllCommitted, ChangeKind, ColumnData, CommitAttempt, CompareOp, LakeColumn, LakePaths,
    LakePredicate, LakeSchema, LakeValue, OperationKind, TimeTravelSpec, TransactionLog,
    append_rows, change_row_counts, changed_ordinals, changes_between, delete_where,
    is_rewrite_only, manifest_as_of, optimize, update_where,
};

const TABLE: u64 = 21;

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
                name: "v".into(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            },
        ],
    )
    .expect("schema")
}

fn attempt(operation: OperationKind, timestamp_us: i64) -> CommitAttempt<'static> {
    CommitAttempt {
        operation,
        db_txn_id: 0,
        commit_lsn: 1,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
        deadline: None,
    }
}

fn rows(ids: &[i64]) -> Vec<ColumnData> {
    vec![
        ColumnData::from_cells(
            0,
            ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
        ),
        ColumnData::from_cells(
            1,
            ids.iter()
                .map(|v| Some((v * 10).to_le_bytes().to_vec()))
                .collect(),
        ),
    ]
}

fn new_log(dir: &std::path::Path) -> TransactionLog {
    TransactionLog::create(
        LakePaths::new(dir, TABLE as u32),
        attempt(OperationKind::SchemaChange, 100),
        &schema(),
        None,
        &BTreeMap::new(),
    )
    .expect("create")
}

fn id_below(limit: i64) -> LakePredicate {
    LakePredicate::Compare {
        column_id: 0,
        op: CompareOp::Lt,
        value: LakeValue::Int(limit),
    }
}

#[test]
fn every_kind_of_commit_reports_the_rows_it_changed() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(
        &log,
        attempt(OperationKind::Append, 200),
        TABLE,
        &rows(&[1, 2, 3, 40, 50]),
    )
    .expect("append");
    delete_where(
        &log,
        attempt(OperationKind::Delete, 300),
        &id_below(3),
        "id < 3",
    )
    .expect("delete");
    // The two rows still live below 41 are rewritten with new images
    update_where(
        &log,
        attempt(OperationKind::Update, 400),
        TABLE,
        Some(&id_below(41)),
        "id < 41",
        &rows(&[3, 40]),
        2,
    )
    .expect("update");

    let changes = changes_between(&log, 1, u64::MAX).expect("changes");
    let by_version: Vec<(u64, ChangeKind, usize)> = changes
        .iter()
        .map(|c| {
            (
                c.version,
                c.kind,
                changed_ordinals(&log, c).expect("rows").len(),
            )
        })
        .collect();
    // Version 2 inserted five rows, version 3 deleted two through its
    // predicate, version 4 rewrote the rows below 41 as a delete of the two
    // that were still live below it and an insert of their new images
    assert!(
        by_version.contains(&(2, ChangeKind::Insert, 5)),
        "{by_version:?}"
    );
    assert!(
        by_version.contains(&(3, ChangeKind::Delete, 2)),
        "{by_version:?}"
    );
    let (deleted, inserted): (Vec<_>, Vec<_>) = by_version
        .iter()
        .filter(|(v, _, _)| *v == 4)
        .partition(|(_, kind, _)| *kind == ChangeKind::Delete);
    assert_eq!(
        deleted.iter().map(|(_, _, n)| n).sum::<usize>(),
        2,
        "{by_version:?}"
    );
    assert_eq!(
        inserted.iter().map(|(_, _, n)| n).sum::<usize>(),
        2,
        "{by_version:?}"
    );
    assert_eq!(change_row_counts(&log, 3, 3, true).expect("counts"), (0, 2));
    assert!(changes.iter().all(|c| c.timestamp_us > 0));
}

#[test]
fn a_predicate_delete_yields_only_the_rows_it_matched() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    append_rows(
        &log,
        attempt(OperationKind::Append, 200),
        TABLE,
        &rows(&[1, 2, 3, 40, 50]),
    )
    .expect("append");
    delete_where(
        &log,
        attempt(OperationKind::Delete, 300),
        &id_below(10),
        "id < 10",
    )
    .expect("delete");
    let changes = changes_between(&log, 3, 3).expect("changes");
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].kind, ChangeKind::Delete);
    assert!(
        changes[0].predicate.is_some(),
        "the file stays, the predicate names the rows"
    );
    let ordinals = changed_ordinals(&log, &changes[0]).expect("rows");
    assert_eq!(ordinals.len(), 3, "the three rows below ten and no other");

    // A second delete over rows the first already removed reports nothing
    // twice
    delete_where(
        &log,
        attempt(OperationKind::Delete, 400),
        &id_below(45),
        "id < 45",
    )
    .expect("delete again");
    let again = changes_between(&log, 4, 4).expect("changes");
    let rows_again: usize = again
        .iter()
        .map(|c| changed_ordinals(&log, c).expect("rows").len())
        .sum();
    assert_eq!(rows_again, 1, "only the row that was still live below 45");
}

#[test]
fn a_compaction_produces_no_change_records() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    // Several small files, which is what a compaction merges
    for batch in 0..4i64 {
        let ids: Vec<i64> = (0..5).map(|i| batch * 100 + i).collect();
        append_rows(
            &log,
            attempt(OperationKind::Append, 200 + batch),
            TABLE,
            &rows(&ids),
        )
        .expect("append");
    }
    let before = log.latest_version();
    let outcome = optimize(
        &log,
        attempt(OperationKind::Optimize, 900),
        TABLE,
        1_000_000,
    )
    .expect("optimize");
    let compacted = outcome.version.expect("the small files were merged");
    assert!(outcome.files_removed >= 2);
    assert_eq!(compacted, before + 1);
    assert!(is_rewrite_only(OperationKind::Optimize));

    let changes = changes_between(&log, compacted, compacted).expect("changes");
    assert!(
        changes.is_empty(),
        "a rewrite that changed no row yields no change: {changes:?}"
    );
    assert_eq!(
        change_row_counts(&log, compacted, compacted, true).expect("counts"),
        (0, 0)
    );

    // The rows are still exactly the ones the appends inserted
    let all = changes_between(&log, 1, u64::MAX).expect("changes");
    let inserted: usize = all
        .iter()
        .filter(|c| c.kind == ChangeKind::Insert)
        .map(|c| changed_ordinals(&log, c).expect("rows").len())
        .sum();
    assert_eq!(inserted, 20);
    assert!(all.iter().all(|c| c.version <= before));
}

#[test]
fn a_change_read_goes_exactly_as_far_back_as_time_travel() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let log = new_log(dir.path());
    for batch in 0..4i64 {
        append_rows(
            &log,
            attempt(OperationKind::Append, 200 + batch),
            TABLE,
            &rows(&[batch]),
        )
        .expect("append");
    }
    assert_eq!(log.latest_version(), 5);
    log.checkpoint(3).expect("checkpoint");
    let removed = log.gc_versions(3).expect("gc");
    assert_eq!(removed, 3, "versions one to three are reclaimed");

    let reopened = TransactionLog::open(LakePaths::new(dir.path(), TABLE as u32), &AllCommitted)
        .expect("reopen");
    // A stream at position p reads the changes above p. It is stale exactly
    // when AS OF p fails. The oldest version time travel still stands at is
    // the oldest position that is not stale, and the changes above it are
    // the oldest a read still answers
    let mut oldest_travel = None;
    let mut oldest_change = None;
    for version in 1..=reopened.latest_version() {
        if oldest_travel.is_none()
            && manifest_as_of(&reopened, TimeTravelSpec::Version(version)).is_ok()
        {
            oldest_travel = Some(version);
        }
        // Version one creates the table and changes nothing, so the search
        // for the oldest change starts above it
        if version >= 2
            && oldest_change.is_none()
            && changes_between(&reopened, version, version).is_ok()
        {
            oldest_change = Some(version);
        }
    }
    assert_eq!(oldest_travel, Some(3), "the checkpoint stands at three");
    assert_eq!(
        oldest_change,
        Some(4),
        "the first change above the oldest position"
    );
    for position in 1..=reopened.latest_version() {
        let travels = manifest_as_of(&reopened, TimeTravelSpec::Version(position)).is_ok();
        let reads = changes_between(&reopened, position + 1, position + 1).is_ok();
        assert_eq!(
            travels, reads,
            "position {position}: time travel {travels}, the changes above it {reads}"
        );
    }
    let live = changes_between(&reopened, 5, 5).expect("the newest change reads");
    assert_eq!(live.len(), 1);
}

//! Change feed over a version range.
//!
//! The transaction log is the change record, so a change query is a replay
//! of version entries rather than a second copy of the data. A version that
//! added a file inserted its rows; a version that removed one deleted the
//! rows that were still live in it; a version that recorded a delete
//! predicate deleted the rows it matches in the files it attached to.
//!
//! An update is a remove plus an add in one commit, so it reports as a
//! delete of the old images and an insert of the new ones. The commit's
//! operation kind travels with every descriptor, so a consumer that needs
//! to distinguish an UPDATE from a DELETE has it without pairing rows that
//! the format never claimed were paired.
//!
//! Descriptors carry no row data and cost one version-file read each. Rows
//! are materialized only for the descriptors a caller asks about, so a
//! change query over a wide range does not decode files nobody reads.

use zyron_common::ZyronError;

use crate::manifest::ManifestFile;
use crate::paths::LakePaths;
use crate::predicate::LakePredicate;
use crate::reader::LakeFileReader;
use crate::transaction_log::{LogEntry, OperationKind, TransactionLog, VersionFileData};

/// Which side of a change a descriptor describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// Rows that did not exist before this version
    Insert,
    /// Rows that were live before this version and are not after it
    Delete,
}

/// One file's worth of change in one version.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangeDescriptor {
    pub version: u64,
    /// The version this change is measured against, always `version - 1`
    pub base_version: u64,
    pub timestamp_us: i64,
    /// The database transaction the commit ran under, zero when standalone
    pub db_txn_id: u64,
    /// The commit's operation, which is what separates an UPDATE's delete
    /// side from a plain DELETE
    pub operation: OperationKind,
    pub kind: ChangeKind,
    pub partition_id: u64,
    /// Present when only the rows this predicate matches changed. None
    /// means every row the file contributed at `base_version`.
    pub predicate: Option<LakePredicate>,
}

/// Every change in versions `from..=to`, in version then entry order.
///
/// Version one creates the table and changes nothing. A range that starts
/// below one starts at one, and a range that ends above the published head
/// ends at the head, so a caller polling for new changes can pass a large
/// upper bound without probing for it first.
pub fn changes_between(
    log: &TransactionLog,
    from: u64,
    to: u64,
) -> Result<Vec<ChangeDescriptor>, ZyronError> {
    let head = log.latest_version();
    let first = from.max(2);
    let last = to.min(head);
    if head == 0 || first > last {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for version in first..=last {
        let path = log.paths().version_file(version);
        let bytes = std::fs::read(&path)?;
        let data = VersionFileData::decode(&bytes, &path.to_string_lossy())?;
        let base_version = version - 1;
        let timestamp_us = data.header.timestamp_us;
        let operation = data.header.operation;
        let db_txn_id = data.header.db_txn_id;

        // The state after this commit, used to find which files a recorded
        // predicate attached to. Only read when the version records one
        let mut after: Option<std::sync::Arc<ManifestFile>> = None;

        for entry in &data.entries {
            match entry {
                LogEntry::AddFile(file) => out.push(ChangeDescriptor {
                    version,
                    base_version,
                    timestamp_us,
                    db_txn_id,
                    operation,
                    kind: ChangeKind::Insert,
                    partition_id: file.partition_id,
                    predicate: None,
                }),
                LogEntry::RemoveFile { partition_id } => out.push(ChangeDescriptor {
                    version,
                    base_version,
                    timestamp_us,
                    db_txn_id,
                    operation,
                    kind: ChangeKind::Delete,
                    partition_id: *partition_id,
                    predicate: None,
                }),
                // An index file holds no rows of its own, it addresses rows
                // the data files already reported. Reporting one here would
                // duplicate every indexed row in the feed
                LogEntry::AddIndex(_)
                | LogEntry::DropIndex { .. }
                | LogEntry::AddIndexFile(_)
                | LogEntry::RemoveIndexFile { .. } => {}
                LogEntry::AddDeletePredicate(del) => {
                    if after.is_none() {
                        after = Some(log.manifest_at(version)?);
                    }
                    let Some(manifest) = &after else { continue };
                    // Files still live after the commit that carry this
                    // predicate. A file the same commit removed is already
                    // reported by its RemoveFile, so it is not double counted
                    for file in &manifest.entries {
                        if file.delete_predicate_ids.contains(&del.id) {
                            out.push(ChangeDescriptor {
                                version,
                                base_version,
                                timestamp_us,
                                db_txn_id,
                                operation,
                                kind: ChangeKind::Delete,
                                partition_id: file.partition_id,
                                predicate: Some(del.predicate.clone()),
                            });
                        }
                    }
                }
                LogEntry::RemoveDeletePredicate { .. }
                | LogEntry::SchemaChange(_)
                | LogEntry::SetClusterSpec(_)
                | LogEntry::SetProperty { .. } => {}
            }
        }
    }
    Ok(out)
}

/// The row ordinals one descriptor covers, ascending.
///
/// An insert covers every row of the added file. A delete covers the rows
/// that were still live at the base version, narrowed by the descriptor's
/// predicate when it has one, so a row already removed by an earlier delete
/// is never reported as deleted twice.
pub fn changed_ordinals(
    log: &TransactionLog,
    descriptor: &ChangeDescriptor,
) -> Result<Vec<u64>, ZyronError> {
    let reader = LakeFileReader::open(log.paths(), descriptor.partition_id)?;
    let row_count = reader.row_count();
    if row_count == 0 {
        return Ok(Vec::new());
    }

    if descriptor.kind == ChangeKind::Insert {
        return Ok((0..row_count as u64).collect());
    }

    // The file as it stood before the commit, which is what "was live" means
    let base = log.manifest_at(descriptor.base_version)?;
    let Some(entry) = base.entry_for(descriptor.partition_id) else {
        return Err(ZyronError::Internal(format!(
            "change feed: partition {:#x} is not in version {}",
            descriptor.partition_id, descriptor.base_version
        )));
    };
    let keep = reader.delete_survivors(&base.schema, &base, entry)?;

    let matched = match &descriptor.predicate {
        None => None,
        Some(predicate) => {
            let columns = reader.read_predicate_columns(&base.schema, &[predicate])?;
            let compiled = crate::reader::CompiledPredicate::new(predicate, &columns);
            Some((columns, compiled))
        }
    };

    let mut out = Vec::new();
    for row in 0..row_count {
        if keep[row / 8] & (1 << (row % 8)) == 0 {
            continue;
        }
        if let Some((columns, compiled)) = &matched {
            if compiled.evaluate(columns, row) != Some(true) {
                continue;
            }
        }
        out.push(row as u64);
    }
    Ok(out)
}

/// Rows inserted and deleted across a version range, materialized exactly.
///
/// This reads every file a change touches, which is the cost of an exact
/// count. The per-version totals in a commit header are the cheap estimate
/// and they count a file's whole row count rather than its live rows.
pub fn change_row_counts(
    log: &TransactionLog,
    from: u64,
    to: u64,
) -> Result<(u64, u64), ZyronError> {
    let mut inserted = 0u64;
    let mut deleted = 0u64;
    for descriptor in changes_between(log, from, to)? {
        let rows = changed_ordinals(log, &descriptor)?.len() as u64;
        match descriptor.kind {
            ChangeKind::Insert => inserted += rows,
            ChangeKind::Delete => deleted += rows,
        }
    }
    Ok((inserted, deleted))
}

/// The data file path a descriptor reads, for a caller materializing rows.
pub fn descriptor_file(paths: &LakePaths, descriptor: &ChangeDescriptor) -> std::path::PathBuf {
    paths.data_file(descriptor.partition_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{append_rows, delete_where, update_where};
    use crate::predicate::{CompareOp, LakeValue};
    use crate::schema::{LakeColumn, LakeSchema};
    use crate::transaction_log::CommitAttempt;
    use crate::writer::ColumnData;
    use std::collections::BTreeMap;
    use zyron_common::TypeId;

    fn schema() -> LakeSchema {
        LakeSchema::new(
            1,
            vec![LakeColumn {
                id: 0,
                name: "id".into(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            }],
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
        vec![ColumnData {
            column_id: 0,
            cells: ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
        }]
    }

    fn new_log(dir: &std::path::Path) -> TransactionLog {
        TransactionLog::create(
            LakePaths::new(dir, 11),
            attempt(OperationKind::SchemaChange, 100),
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create")
    }

    #[test]
    fn test_append_reports_every_row_as_an_insert() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            11,
            &rows(&[1, 2, 3]),
        )
        .expect("append");

        let changes = changes_between(&log, 1, 99).expect("changes");
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].version, 2);
        assert_eq!(changes[0].kind, ChangeKind::Insert);
        assert_eq!(changes[0].operation, OperationKind::Append);
        assert_eq!(changes[0].timestamp_us, 200);
        assert_eq!(changed_ordinals(&log, &changes[0]).expect("rows").len(), 3);

        // Version one creates the table and changes nothing
        assert!(changes_between(&log, 1, 1).expect("changes").is_empty());
        assert_eq!(change_row_counts(&log, 1, 99).expect("counts"), (3, 0));
    }

    #[test]
    fn test_whole_file_delete_reports_the_rows_that_were_live() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            11,
            &rows(&[1, 2, 3]),
        )
        .expect("append");
        delete_where(
            &log,
            attempt(OperationKind::Delete, 300),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(100),
            },
            "id < 100",
        )
        .expect("delete");

        let changes = changes_between(&log, 3, 3).expect("changes");
        assert_eq!(changes.len(), 1, "the covered file drops whole");
        assert_eq!(changes[0].kind, ChangeKind::Delete);
        assert!(changes[0].predicate.is_none());
        assert_eq!(changed_ordinals(&log, &changes[0]).expect("rows").len(), 3);
        assert_eq!(change_row_counts(&log, 3, 3).expect("counts"), (0, 3));
    }

    #[test]
    fn test_partial_delete_reports_only_the_matching_rows() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            11,
            &rows(&[1, 2, 3, 40, 50]),
        )
        .expect("append");
        delete_where(
            &log,
            attempt(OperationKind::Delete, 300),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(10),
            },
            "id < 10",
        )
        .expect("delete");

        let changes = changes_between(&log, 3, 3).expect("changes");
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].kind, ChangeKind::Delete);
        assert!(
            changes[0].predicate.is_some(),
            "a partial delete records one"
        );
        assert_eq!(
            changed_ordinals(&log, &changes[0]).expect("rows").len(),
            3,
            "only the three rows below ten"
        );
        assert_eq!(change_row_counts(&log, 3, 3).expect("counts"), (0, 3));
    }

    #[test]
    fn test_a_row_already_deleted_is_not_reported_deleted_twice() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            11,
            &rows(&[1, 2, 3, 40, 50]),
        )
        .expect("append");
        delete_where(
            &log,
            attempt(OperationKind::Delete, 300),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(3),
            },
            "id < 3",
        )
        .expect("first delete");
        delete_where(
            &log,
            attempt(OperationKind::Delete, 400),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(45),
            },
            "id < 45",
        )
        .expect("second delete");

        // The second delete's predicate also matches ids 1 and 2, which the
        // first delete already removed, so only 3 and 40 are new deletions
        let (inserted, deleted) = change_row_counts(&log, 4, 4).expect("counts");
        assert_eq!(inserted, 0);
        assert_eq!(deleted, 2);
        // Across the whole history every row is inserted once and the four
        // matching rows are deleted once each
        assert_eq!(change_row_counts(&log, 1, 99).expect("counts"), (5, 4));
    }

    #[test]
    fn test_update_reports_a_delete_and_an_insert_under_one_version() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            11,
            &rows(&[1, 2, 9]),
        )
        .expect("append");
        let predicate = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Lt,
            value: LakeValue::Int(5),
        };
        update_where(
            &log,
            attempt(OperationKind::Update, 300),
            11,
            Some(&predicate),
            "id < 5",
            &rows(&[101, 102]),
            2,
        )
        .expect("update");

        let changes = changes_between(&log, 3, 3).expect("changes");
        assert!(
            changes.iter().all(|c| c.operation == OperationKind::Update),
            "the commit kind travels with every descriptor"
        );
        assert_eq!(
            changes
                .iter()
                .filter(|c| c.kind == ChangeKind::Insert)
                .count(),
            1
        );
        assert!(changes.iter().any(|c| c.kind == ChangeKind::Delete));

        let (inserted, deleted) = change_row_counts(&log, 3, 3).expect("counts");
        assert_eq!(inserted, 2, "the two new images");
        assert_eq!(deleted, 2, "the two old images");
    }

    #[test]
    fn test_range_is_clamped_to_the_published_head() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 11, &rows(&[1])).expect("append");

        assert_eq!(
            changes_between(&log, 0, u64::MAX).expect("changes").len(),
            1
        );
        assert!(changes_between(&log, 5, 9).expect("changes").is_empty());
        assert!(changes_between(&log, 3, 2).expect("changes").is_empty());
    }
}

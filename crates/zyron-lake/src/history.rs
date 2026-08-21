//! Version query API over a table's transaction log.
//!
//! Every commit a table takes is a numbered version file whose 128-byte
//! header is a complete audit record, so the history of a table is readable
//! without touching one byte of data. A history listing costs one positioned
//! read per version, a version's file list costs the manifest at that
//! version, and a diff is a set difference between two manifests.
//!
//! Everything here reads published versions only. A version still pending
//! its database transaction is not part of the table's history yet.

use std::collections::BTreeMap;

use zyron_common::ZyronError;

use crate::manifest::PartitionEntry;
use crate::schema::LakeSchema;
use crate::transaction_log::{
    CommitInfo, LogEntry, OperationKind, TransactionLog, VersionFileData, read_commit_header,
};

/// One row of a table's version history.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionRecord {
    pub version: u64,
    /// Version this commit was built against, its parent in the lineage
    pub read_version: u64,
    pub operation: OperationKind,
    pub timestamp_us: i64,
    /// Enclosing database transaction, zero for a standalone commit
    pub db_txn_id: u64,
    pub commit_lsn: u64,
    pub files_added: u32,
    pub files_removed: u32,
    pub rows_added: u64,
    pub rows_removed: u64,
    pub bytes_added: u64,
}

/// One version with its audit block and the entries it carried.
#[derive(Debug, Clone)]
pub struct VersionDetails {
    pub record: VersionRecord,
    /// Identity, client and annotation the writer attached, when it did
    pub audit: Option<CommitInfo>,
    /// Data files this version added
    pub files_added: Vec<PartitionEntry>,
    /// Data file ids this version removed
    pub files_removed: Vec<u64>,
    /// Delete predicates this version recorded, as SQL text
    pub delete_predicates: Vec<String>,
    /// Properties this version set
    pub properties: BTreeMap<String, String>,
    /// Set when this version replaced the schema
    pub schema_id: Option<u64>,
    /// Index maintenance this version performed, one line per change.
    /// Index files are kept out of `files_added` and `files_removed`,
    /// which report the table's own data, so a version that only rebuilt
    /// an index does not read as one that rewrote rows
    pub index_changes: Vec<String>,
}

/// What changed in the live file set between two versions.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionDiff {
    pub from_version: u64,
    pub to_version: u64,
    /// Files live at `to` that were not live at `from`
    pub files_added: Vec<u64>,
    /// Files live at `from` that are no longer live at `to`
    pub files_removed: Vec<u64>,
    /// Rows in the added files
    pub rows_added: u64,
    /// Rows in the removed files, which counts a rewritten file's rows on
    /// both sides because an optimize removes and adds at once
    pub rows_removed: u64,
    pub bytes_added: u64,
    pub bytes_removed: u64,
}

/// Reads the header of one published version.
fn record_at(log: &TransactionLog, version: u64) -> Result<VersionRecord, ZyronError> {
    let header = read_commit_header(&log.paths().version_file(version))?;
    Ok(VersionRecord {
        version: header.version,
        read_version: header.read_version,
        operation: header.operation,
        timestamp_us: header.timestamp_us,
        db_txn_id: header.db_txn_id,
        commit_lsn: header.commit_lsn,
        files_added: header.files_added,
        files_removed: header.files_removed,
        rows_added: header.rows_added,
        rows_removed: header.rows_removed,
        bytes_added: header.bytes_added,
    })
}

/// The table's history, newest version first.
///
/// `limit` bounds the walk so a long lived table does not read every version
/// file to answer a question about its last few commits. Zero returns
/// nothing rather than everything, which is what a caller asking for zero
/// rows means.
pub fn table_history(log: &TransactionLog, limit: usize) -> Result<Vec<VersionRecord>, ZyronError> {
    let head = log.latest_version();
    if head == 0 || limit == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(limit.min(head as usize));
    let mut version = head;
    loop {
        out.push(record_at(log, version)?);
        if out.len() >= limit || version == 1 {
            break;
        }
        version -= 1;
    }
    Ok(out)
}

/// One version with its audit block and entry breakdown.
pub fn version_details(log: &TransactionLog, version: u64) -> Result<VersionDetails, ZyronError> {
    check_published(log, version)?;
    let path = log.paths().version_file(version);
    let bytes = std::fs::read(&path)?;
    let data = VersionFileData::decode(&bytes, &path.to_string_lossy())?;

    let mut files_added = Vec::new();
    let mut files_removed = Vec::new();
    let mut delete_predicates = Vec::new();
    let mut properties = BTreeMap::new();
    let mut schema_id = None;
    let mut index_changes = Vec::new();
    for entry in &data.entries {
        match entry {
            LogEntry::AddFile(file) => files_added.push(file.clone()),
            LogEntry::RemoveFile { partition_id } => files_removed.push(*partition_id),
            LogEntry::AddDeletePredicate(del) => delete_predicates.push(del.sql.clone()),
            LogEntry::RemoveDeletePredicate { .. } => {}
            LogEntry::SchemaChange(schema) => schema_id = Some(schema.schema_id),
            LogEntry::SetClusterSpec(_) => {}
            LogEntry::SetProperty { key, value } => {
                properties.insert(key.clone(), value.clone());
            }
            LogEntry::AddIndex(spec) => {
                index_changes.push(format!("create index {}", spec.name));
            }
            LogEntry::DropIndex { index_id } => {
                index_changes.push(format!("drop index {}", index_id));
            }
            LogEntry::AddIndexFile(file) => index_changes.push(format!(
                "index {} add file {:#x} covering {} partitions",
                file.index_id,
                file.file.partition_id,
                file.covers.len()
            )),
            LogEntry::RemoveIndexFile {
                index_id,
                partition_id,
            } => index_changes.push(format!(
                "index {} remove file {:#x}",
                index_id, partition_id
            )),
        }
    }

    let h = data.header;
    Ok(VersionDetails {
        record: VersionRecord {
            version: h.version,
            read_version: h.read_version,
            operation: h.operation,
            timestamp_us: h.timestamp_us,
            db_txn_id: h.db_txn_id,
            commit_lsn: h.commit_lsn,
            files_added: h.files_added,
            files_removed: h.files_removed,
            rows_added: h.rows_added,
            rows_removed: h.rows_removed,
            bytes_added: h.bytes_added,
        },
        audit: data.audit,
        files_added,
        files_removed,
        delete_predicates,
        properties,
        schema_id,
        index_changes,
    })
}

/// Every data file live at one version, in manifest order.
pub fn version_files(
    log: &TransactionLog,
    version: u64,
) -> Result<Vec<PartitionEntry>, ZyronError> {
    check_published(log, version)?;
    Ok(log.manifest_at(version)?.entries.clone())
}

/// The schema in force at one version.
pub fn schema_at_version(log: &TransactionLog, version: u64) -> Result<LakeSchema, ZyronError> {
    check_published(log, version)?;
    Ok(log.manifest_at(version)?.schema.clone())
}

/// The live file sets of two versions, differenced.
pub fn diff_versions(
    log: &TransactionLog,
    from_version: u64,
    to_version: u64,
) -> Result<VersionDiff, ZyronError> {
    check_published(log, from_version)?;
    check_published(log, to_version)?;
    let from = log.manifest_at(from_version)?;
    let to = log.manifest_at(to_version)?;

    let mut diff = VersionDiff {
        from_version,
        to_version,
        files_added: Vec::new(),
        files_removed: Vec::new(),
        rows_added: 0,
        rows_removed: 0,
        bytes_added: 0,
        bytes_removed: 0,
    };
    // Manifest entries are sorted by partition id, so membership is a
    // binary search rather than a set build
    for entry in &to.entries {
        if from.entry_for(entry.partition_id).is_none() {
            diff.files_added.push(entry.partition_id);
            diff.rows_added += entry.row_count;
            diff.bytes_added += entry.size_bytes;
        }
    }
    for entry in &from.entries {
        if to.entry_for(entry.partition_id).is_none() {
            diff.files_removed.push(entry.partition_id);
            diff.rows_removed += entry.row_count;
            diff.bytes_removed += entry.size_bytes;
        }
    }
    Ok(diff)
}

/// The chain of versions one version was built on, newest first.
///
/// Each commit records the version it read, so following that link back
/// gives the writers that actually serialized against each other rather
/// than the plain numeric sequence. A concurrent append that won the race
/// against several others appears once, and the versions it skipped are the
/// ones whose work it never saw.
pub fn version_lineage(log: &TransactionLog, version: u64) -> Result<Vec<u64>, ZyronError> {
    check_published(log, version)?;
    let mut chain = vec![version];
    let mut current = version;
    while current > 1 {
        let header = read_commit_header(&log.paths().version_file(current))?;
        let parent = header.read_version;
        if parent == 0 || parent >= current {
            break;
        }
        chain.push(parent);
        current = parent;
    }
    Ok(chain)
}

fn check_published(log: &TransactionLog, version: u64) -> Result<(), ZyronError> {
    let head = log.latest_version();
    if version == 0 || version > head {
        return Err(ZyronError::VersionNotFound(version));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{append_rows, delete_where};
    use crate::paths::LakePaths;
    use crate::predicate::{CompareOp, LakePredicate, LakeValue};
    use crate::schema::LakeColumn;
    use crate::transaction_log::CommitAttempt;
    use crate::writer::ColumnData;
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
            commit_lsn: 5,
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

    /// Builds a log with four versions: create, two appends, one delete.
    fn populated(dir: &std::path::Path) -> TransactionLog {
        let log = TransactionLog::create(
            LakePaths::new(dir, 9),
            attempt(OperationKind::SchemaChange, 1_000),
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create");
        append_rows(
            &log,
            attempt(OperationKind::Append, 2_000),
            9,
            &rows(&[1, 2, 3]),
        )
        .expect("append one");
        append_rows(
            &log,
            attempt(OperationKind::Append, 3_000),
            9,
            &rows(&[10, 11]),
        )
        .expect("append two");
        delete_where(
            &log,
            attempt(OperationKind::Delete, 4_000),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(5),
            },
            "id < 5",
        )
        .expect("delete");
        log
    }

    #[test]
    fn test_history_reports_every_commit_newest_first() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = populated(dir.path());

        let all = table_history(&log, usize::MAX).expect("history");
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].version, 4);
        assert_eq!(all[0].operation, OperationKind::Delete);
        assert_eq!(all[3].version, 1);
        assert_eq!(all[3].operation, OperationKind::SchemaChange);
        assert_eq!(all[1].operation, OperationKind::Append);
        assert_eq!(all[1].rows_added, 2);
        assert_eq!(all[2].rows_added, 3);
        assert_eq!(all[0].timestamp_us, 4_000);

        // The limit bounds the walk instead of reading every version
        let recent = table_history(&log, 2).expect("history");
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].version, 4);
        assert_eq!(recent[1].version, 3);
        assert!(table_history(&log, 0).expect("history").is_empty());
    }

    #[test]
    fn test_version_details_breaks_out_the_entries() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = populated(dir.path());

        let append = version_details(&log, 2).expect("details");
        assert_eq!(append.record.operation, OperationKind::Append);
        assert_eq!(append.files_added.len(), 1);
        assert_eq!(append.files_added[0].row_count, 3);
        assert!(append.files_removed.is_empty());
        assert!(append.delete_predicates.is_empty());

        let delete = version_details(&log, 4).expect("details");
        assert_eq!(delete.record.operation, OperationKind::Delete);
        // The whole first file is covered by id < 5, so it drops outright
        assert_eq!(delete.files_removed.len(), 1);

        let created = version_details(&log, 1).expect("details");
        assert_eq!(created.schema_id, Some(1));
    }

    #[test]
    fn test_version_files_and_schema_read_the_past() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = populated(dir.path());

        assert_eq!(version_files(&log, 1).expect("files").len(), 0);
        assert_eq!(version_files(&log, 2).expect("files").len(), 1);
        assert_eq!(version_files(&log, 3).expect("files").len(), 2);
        // The delete dropped the covered file whole
        assert_eq!(version_files(&log, 4).expect("files").len(), 1);

        assert_eq!(schema_at_version(&log, 3).expect("schema").schema_id, 1);
        assert!(matches!(
            schema_at_version(&log, 99),
            Err(ZyronError::VersionNotFound(99))
        ));
    }

    #[test]
    fn test_diff_versions_is_a_live_file_set_difference() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = populated(dir.path());

        let grew = diff_versions(&log, 2, 3).expect("diff");
        assert_eq!(grew.files_added.len(), 1);
        assert_eq!(grew.rows_added, 2);
        assert!(grew.files_removed.is_empty());
        assert_eq!(grew.rows_removed, 0);
        assert!(grew.bytes_added > 0);

        let shrank = diff_versions(&log, 3, 4).expect("diff");
        assert!(shrank.files_added.is_empty());
        assert_eq!(shrank.files_removed.len(), 1);
        assert_eq!(shrank.rows_removed, 3);

        // A version against itself is the empty diff
        let same = diff_versions(&log, 3, 3).expect("diff");
        assert!(same.files_added.is_empty() && same.files_removed.is_empty());
    }

    #[test]
    fn test_version_lineage_follows_the_read_version_links() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = populated(dir.path());

        // Serial commits each read the version before them
        assert_eq!(version_lineage(&log, 4).expect("lineage"), vec![4, 3, 2, 1]);
        assert_eq!(version_lineage(&log, 2).expect("lineage"), vec![2, 1]);
        assert_eq!(version_lineage(&log, 1).expect("lineage"), vec![1]);
        assert!(matches!(
            version_lineage(&log, 0),
            Err(ZyronError::VersionNotFound(0))
        ));
    }
}

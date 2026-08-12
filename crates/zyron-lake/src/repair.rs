//! Validation, repair and orphan cleanup.
//!
//! Every check here reads metadata only: version headers, entry sections and
//! the .zyr file headers a manifest points at. Nothing decodes a column, so
//! validating a petabyte table costs a few thousand small reads.
//!
//! Repair never invents data. It removes state that is provably unreachable
//! (a corrupt checkpoint the versions can replay past, a version tail after a
//! gap) and, only when the caller asks for it, drops manifest entries whose
//! data file is gone. Dropping those loses rows, so it is opt in and always
//! reported rather than applied quietly.

use std::collections::BTreeSet;
use std::fs;

use zyron_common::ZyronError;

use crate::branch::list_branches;
use crate::paths::{
    parse_data_file_name, parse_index_file_name, parse_version_file_name, LakePaths,
    VersionFileKind,
};
use crate::reader::LakeFileReader;
use crate::transaction_log::{CommitAttempt, LogEntry, TransactionLog, VersionFileData};

/// One thing wrong with a table's on-disk state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    /// A version in the published range has no file
    MissingVersion { version: u64 },
    /// A version file does not decode, or disagrees with its own name
    UnreadableVersion { version: u64, reason: String },
    /// A checkpoint does not decode
    UnreadableCheckpoint { version: u64, reason: String },
    /// A checkpoint's file set differs from replaying the log to that version
    CheckpointDiverged { version: u64, reason: String },
    /// The manifest references a data file that is not on disk
    MissingDataFile { partition_id: u64 },
    /// A data file's row count differs from what the manifest recorded
    RowCountMismatch {
        partition_id: u64,
        manifest_rows: u64,
        file_rows: u64,
    },
    /// A file carries a delete predicate id the manifest does not define
    DanglingDeletePredicate { partition_id: u64, predicate_id: u64 },
}

impl Problem {
    /// True when repair can clear this without losing rows.
    pub fn is_repairable_without_data_loss(&self) -> bool {
        matches!(
            self,
            Problem::UnreadableCheckpoint { .. } | Problem::CheckpointDiverged { .. }
        )
    }
}

/// What validation found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationReport {
    pub head_version: u64,
    pub versions_checked: u64,
    pub checkpoints_checked: u64,
    pub files_checked: u64,
    pub problems: Vec<Problem>,
}

impl ValidationReport {
    pub fn is_healthy(&self) -> bool {
        self.problems.is_empty()
    }
}

/// Checks a table's log, checkpoints and data files against each other.
///
/// `deep` additionally replays every checkpoint's version and compares the
/// file set, which is the only way a silently diverged checkpoint shows up.
pub fn validate(log: &TransactionLog, deep: bool) -> Result<ValidationReport, ZyronError> {
    let paths = log.paths();
    let head = log.latest_version();
    let mut report = ValidationReport {
        head_version: head,
        ..Default::default()
    };

    for version in 1..=head {
        let path = log.version_path(version);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                report.problems.push(Problem::MissingVersion { version });
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        report.versions_checked += 1;
        match VersionFileData::decode(&bytes, &path.to_string_lossy()) {
            Ok(data) if data.header.version != version => {
                report.problems.push(Problem::UnreadableVersion {
                    version,
                    reason: format!("header says version {}", data.header.version),
                });
            }
            Ok(_) => {}
            Err(e) => report.problems.push(Problem::UnreadableVersion {
                version,
                reason: e.to_string(),
            }),
        }
    }

    for version in checkpoint_versions(paths)? {
        report.checkpoints_checked += 1;
        let path = paths.checkpoint_file(version);
        let bytes = fs::read(&path)?;
        let checkpoint = match crate::manifest::ManifestFile::decode(&bytes, &path.to_string_lossy())
        {
            Ok(m) => m,
            Err(e) => {
                report.problems.push(Problem::UnreadableCheckpoint {
                    version,
                    reason: e.to_string(),
                });
                continue;
            }
        };
        if deep && version <= head {
            let replayed = log.manifest_at(version)?;
            let a: Vec<u64> = checkpoint.entries.iter().map(|e| e.partition_id).collect();
            let b: Vec<u64> = replayed.entries.iter().map(|e| e.partition_id).collect();
            if a != b {
                report.problems.push(Problem::CheckpointDiverged {
                    version,
                    reason: format!("{} files in the checkpoint, {} replayed", a.len(), b.len()),
                });
            }
        }
    }

    let manifest = log.latest_manifest()?;
    let predicate_ids: BTreeSet<u64> = manifest.delete_predicates.iter().map(|p| p.id).collect();
    for entry in &manifest.entries {
        report.files_checked += 1;
        for id in &entry.delete_predicate_ids {
            if !predicate_ids.contains(id) {
                report.problems.push(Problem::DanglingDeletePredicate {
                    partition_id: entry.partition_id,
                    predicate_id: *id,
                });
            }
        }
        if !paths.data_file(entry.partition_id).exists() {
            report.problems.push(Problem::MissingDataFile {
                partition_id: entry.partition_id,
            });
            continue;
        }
        // Reads the .zyr header and segment index, never a column
        let reader = LakeFileReader::open(paths, entry.partition_id)?;
        let file_rows = reader.row_count() as u64;
        if file_rows != entry.row_count {
            report.problems.push(Problem::RowCountMismatch {
                partition_id: entry.partition_id,
                manifest_rows: entry.row_count,
                file_rows,
            });
        }
    }

    Ok(report)
}

/// How aggressive a repair pass is allowed to be.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepairOptions {
    /// Commit a removal for every manifest entry whose data file is gone.
    ///
    /// Those rows are already unreadable, and until they are removed every
    /// scan of the table fails. Removing them makes the rest readable and
    /// loses the missing file's rows, so it is never done unasked.
    pub remove_missing_files: bool,
}

/// What a repair pass changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepairReport {
    /// Checkpoints deleted, the log replays past them without loss
    pub checkpoints_removed: Vec<u64>,
    /// Version files deleted because a gap made them unreachable
    pub versions_removed: Vec<u64>,
    /// Manifest entries removed for missing data files
    pub files_removed: Vec<u64>,
    /// The version a removal committed, None when nothing was committed
    pub version: Option<u64>,
    /// Problems this pass could not fix
    pub unrepaired: Vec<Problem>,
}

/// Clears what can be cleared and reports what it could not.
pub fn repair(
    log: &TransactionLog,
    options: RepairOptions,
    attempt: CommitAttempt<'_>,
) -> Result<RepairReport, ZyronError> {
    let paths = log.paths();
    let mut report = RepairReport::default();

    // A checkpoint is derived state. Deleting a bad one costs a longer
    // replay and loses nothing
    let before = validate(log, true)?;
    for problem in &before.problems {
        match problem {
            Problem::UnreadableCheckpoint { version, .. }
            | Problem::CheckpointDiverged { version, .. } => {
                let path = paths.checkpoint_file(*version);
                if path.exists() {
                    fs::remove_file(&path)?;
                    report.checkpoints_removed.push(*version);
                }
            }
            _ => {}
        }
    }

    // Version files past the published head are unreachable: the head is
    // the last contiguous committed version, so anything above it was built
    // on state no reader can see
    let head = log.latest_version();
    for version in log_versions(paths)? {
        if version > head {
            let path = paths.version_file(version);
            if path.exists() {
                fs::remove_file(&path)?;
                report.versions_removed.push(version);
            }
        }
    }

    let missing: Vec<u64> = before
        .problems
        .iter()
        .filter_map(|p| match p {
            Problem::MissingDataFile { partition_id } => Some(*partition_id),
            _ => None,
        })
        .collect();
    if !missing.is_empty() {
        if options.remove_missing_files {
            let removed = missing.clone();
            let version = log.commit(attempt, |base| {
                let mut entries = Vec::new();
                for id in &removed {
                    if base.entry_for(*id).is_some() {
                        entries.push(LogEntry::RemoveFile { partition_id: *id });
                    }
                }
                if entries.is_empty() {
                    return Err(ZyronError::Internal(
                        "the missing files are already out of the manifest".into(),
                    ));
                }
                Ok(entries)
            })?;
            report.files_removed = missing;
            report.version = Some(version);
        } else {
            report
                .unrepaired
                .extend(missing.into_iter().map(|partition_id| {
                    Problem::MissingDataFile { partition_id }
                }));
        }
    }

    // Everything else is a data problem no metadata edit can fix
    for problem in before.problems {
        if !problem.is_repairable_without_data_loss()
            && !matches!(problem, Problem::MissingDataFile { .. })
        {
            report.unrepaired.push(problem);
        }
    }

    Ok(report)
}

/// What orphan cleanup found and removed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OrphanReport {
    /// Data files no retained version or branch references
    pub removed: Vec<u64>,
    pub bytes_reclaimed: u64,
    /// Half-written files a crashed writer left staged. Never deleted: an
    /// in-flight write is producing one right now
    pub staged_files: usize,
    /// Index files reclaimed, as index id and partition id. Reported apart
    /// from the data files so a report never reads as row loss
    pub removed_index_files: Vec<(u32, u64)>,
}

/// Deletes data files no retained version and no branch can reach.
///
/// Reachability is the union of the manifest at `retain_from_version` and
/// every file any later version added, on main and on every branch, so a
/// time-travel read inside the retention window keeps working. A file the
/// current manifest dropped but an older retained version still names is
/// not an orphan.
pub fn cleanup_orphans(
    log: &TransactionLog,
    retain_from_version: u64,
) -> Result<OrphanReport, ZyronError> {
    let paths = log.paths();
    let head = log.latest_version();
    let floor = retain_from_version.clamp(1, head.max(1));

    let floor_manifest = log.manifest_at(floor)?;
    let mut reachable: BTreeSet<u64> =
        floor_manifest.entries.iter().map(|e| e.partition_id).collect();
    let mut reachable_index: BTreeSet<(u32, u64)> = floor_manifest
        .index_files
        .iter()
        .map(|f| (f.index_id, f.file.partition_id))
        .collect();
    collect_added(log, floor + 1, head, &mut reachable, &mut reachable_index)?;

    for info in list_branches(paths)? {
        let branch = TransactionLog::open_branch(paths.clone(), &info.name, info.base_version)?;
        // Everything at or below the fork point is main's history, which the
        // floor above already covers when it is retained. The branch's own
        // versions are what main never saw
        collect_added(
            &branch,
            info.base_version + 1,
            branch.latest_version(),
            &mut reachable,
            &mut reachable_index,
        )?;
        let head_manifest = branch.latest_manifest()?;
        for entry in &head_manifest.entries {
            reachable.insert(entry.partition_id);
        }
        for file in &head_manifest.index_files {
            reachable_index.insert((file.index_id, file.file.partition_id));
        }
    }

    let mut report = OrphanReport::default();
    let dir = paths.data_dir();
    if !dir.exists() {
        return Ok(report);
    }
    for dirent in fs::read_dir(&dir)? {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        // An index file is reachable on its own key, so a dropped index
        // frees its files while a live one keeps them
        if let Some(key) = parse_index_file_name(name) {
            if reachable_index.contains(&key) {
                continue;
            }
            let size = dirent.metadata().map(|m| m.len()).unwrap_or(0);
            fs::remove_file(dirent.path())?;
            report.removed_index_files.push(key);
            report.bytes_reclaimed += size;
            continue;
        }
        let Some(partition_id) = parse_data_file_name(name) else {
            if name.ends_with(".tmp") {
                report.staged_files += 1;
            }
            continue;
        };
        if reachable.contains(&partition_id) {
            continue;
        }
        let size = dirent.metadata().map(|m| m.len()).unwrap_or(0);
        fs::remove_file(dirent.path())?;
        report.removed.push(partition_id);
        report.bytes_reclaimed += size;
    }
    report.removed.sort_unstable();
    Ok(report)
}

/// Adds every file id an Add entry introduced over a version range, data
/// files and index files alike.
fn collect_added(
    log: &TransactionLog,
    from: u64,
    to: u64,
    out: &mut BTreeSet<u64>,
    out_index: &mut BTreeSet<(u32, u64)>,
) -> Result<(), ZyronError> {
    if from > to {
        return Ok(());
    }
    for version in from..=to {
        let path = log.version_path(version);
        let bytes = fs::read(&path)?;
        let data = VersionFileData::decode(&bytes, &path.to_string_lossy())?;
        for entry in &data.entries {
            match entry {
                LogEntry::AddFile(file) => {
                    out.insert(file.partition_id);
                }
                LogEntry::AddIndexFile(file) => {
                    out_index.insert((file.index_id, file.file.partition_id));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn checkpoint_versions(paths: &LakePaths) -> Result<Vec<u64>, ZyronError> {
    scan_log_dir(paths, VersionFileKind::Checkpoint)
}

fn log_versions(paths: &LakePaths) -> Result<Vec<u64>, ZyronError> {
    scan_log_dir(paths, VersionFileKind::Version)
}

fn scan_log_dir(paths: &LakePaths, want: VersionFileKind) -> Result<Vec<u64>, ZyronError> {
    let mut out = Vec::new();
    for dirent in fs::read_dir(paths.log_dir())? {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some((v, kind)) = parse_version_file_name(name) {
            if kind == want {
                out.push(v);
            }
        }
    }
    out.sort_unstable();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{append_rows, delete_where};
    use crate::predicate::{CompareOp, LakePredicate, LakeValue};
    use crate::schema::{LakeColumn, LakeSchema};
    use crate::transaction_log::OperationKind;
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
                ts_precision: None,
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
            audit: None,
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
            LakePaths::new(dir, 17),
            attempt(OperationKind::SchemaChange, 100),
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create")
    }

    fn data_files(paths: &LakePaths) -> Vec<u64> {
        let mut out: Vec<u64> = fs::read_dir(paths.data_dir())
            .expect("data dir")
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                e.file_name()
                    .to_str()
                    .and_then(parse_data_file_name)
            })
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn test_validate_reports_a_healthy_table() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 17, &rows(&[1, 2, 3]))
            .expect("append");
        append_rows(&log, attempt(OperationKind::Append, 300), 17, &rows(&[4, 5]))
            .expect("append");

        let report = validate(&log, true).expect("validate");
        assert!(report.is_healthy(), "{:?}", report.problems);
        assert_eq!(report.head_version, 3);
        assert_eq!(report.versions_checked, 3);
        assert_eq!(report.files_checked, 2);
    }

    #[test]
    fn test_a_missing_data_file_is_reported_and_only_dropped_when_asked() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 17, &rows(&[1, 2, 3]))
            .expect("append");
        append_rows(&log, attempt(OperationKind::Append, 300), 17, &rows(&[4, 5]))
            .expect("append");

        let victim = log.latest_manifest().expect("manifest").entries[0].partition_id;
        fs::remove_file(log.paths().data_file(victim)).expect("remove data file");

        let report = validate(&log, false).expect("validate");
        assert_eq!(
            report.problems,
            vec![Problem::MissingDataFile {
                partition_id: victim
            }]
        );

        // The default refuses to drop rows, it reports what it would take
        let conservative = repair(
            &log,
            RepairOptions::default(),
            attempt(OperationKind::Vacuum, 400),
        )
        .expect("repair");
        assert!(conservative.files_removed.is_empty());
        assert_eq!(conservative.version, None);
        assert_eq!(
            conservative.unrepaired,
            vec![Problem::MissingDataFile {
                partition_id: victim
            }]
        );
        assert_eq!(log.latest_version(), 3, "nothing was committed");

        // Asked explicitly, it removes the entry so the rest reads again
        let applied = repair(
            &log,
            RepairOptions {
                remove_missing_files: true,
            },
            attempt(OperationKind::Vacuum, 500),
        )
        .expect("repair");
        assert_eq!(applied.files_removed, vec![victim]);
        assert_eq!(applied.version, Some(4));
        assert!(validate(&log, true).expect("validate").is_healthy());
    }

    #[test]
    fn test_a_corrupt_checkpoint_is_reported_and_deleted() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 17, &rows(&[1, 2]))
            .expect("append");
        log.checkpoint(2).expect("checkpoint");

        let path = log.paths().checkpoint_file(2);
        assert!(path.exists());
        fs::write(&path, b"not a manifest at all").expect("corrupt it");

        let report = validate(&log, true).expect("validate");
        assert!(matches!(
            report.problems.as_slice(),
            [Problem::UnreadableCheckpoint { version: 2, .. }]
        ));

        // A checkpoint is derived state, deleting it costs a longer replay
        let repaired = repair(
            &log,
            RepairOptions::default(),
            attempt(OperationKind::Vacuum, 300),
        )
        .expect("repair");
        assert_eq!(repaired.checkpoints_removed, vec![2]);
        assert!(!path.exists());
        assert!(validate(&log, true).expect("validate").is_healthy());
        // The table still reads, replayed from version one
        assert_eq!(log.latest_manifest().expect("manifest").entries.len(), 1);
    }

    #[test]
    fn test_cleanup_orphans_respects_the_retention_floor() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 17, &rows(&[1, 2, 3]))
            .expect("append");
        append_rows(&log, attempt(OperationKind::Append, 300), 17, &rows(&[40, 50]))
            .expect("append");
        // Drops the first file whole, leaving it on disk but out of the manifest
        delete_where(
            &log,
            attempt(OperationKind::Delete, 400),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(10),
            },
            "id < 10",
        )
        .expect("delete");
        assert_eq!(data_files(log.paths()).len(), 2);

        // Retaining from version one keeps every file a past version names
        let kept = cleanup_orphans(&log, 1).expect("cleanup");
        assert!(kept.removed.is_empty(), "time travel still needs the file");
        assert_eq!(data_files(log.paths()).len(), 2);

        // Retaining only the head makes the dropped file unreachable
        let head = log.latest_version();
        let cleaned = cleanup_orphans(&log, head).expect("cleanup");
        assert_eq!(cleaned.removed.len(), 1);
        assert!(cleaned.bytes_reclaimed > 0);
        assert_eq!(data_files(log.paths()).len(), 1);
        assert!(validate(&log, true).expect("validate").is_healthy());
    }

    #[test]
    fn test_cleanup_orphans_keeps_what_a_branch_references() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 17, &rows(&[1]))
            .expect("append");
        crate::branch::create_branch(&log, "work", None, 250).expect("branch");

        let branch = crate::branch::open_branch(log.paths(), "work").expect("open");
        append_rows(&branch, attempt(OperationKind::Append, 300), 17, &rows(&[9]))
            .expect("branch append");
        assert_eq!(data_files(log.paths()).len(), 2);

        // The branch's file is not in main's manifest at any version, and it
        // must survive a cleanup that only main's head would justify
        let report = cleanup_orphans(&log, log.latest_version()).expect("cleanup");
        assert!(report.removed.is_empty(), "a branch's file is not an orphan");
        assert_eq!(data_files(log.paths()).len(), 2);

        // Once the branch is gone so is its claim
        crate::branch::drop_branch(log.paths(), "work").expect("drop");
        let report = cleanup_orphans(&log, log.latest_version()).expect("cleanup");
        assert_eq!(report.removed.len(), 1);
        assert_eq!(data_files(log.paths()).len(), 1);
    }

    #[test]
    fn test_staged_files_are_counted_never_deleted() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 17, &rows(&[1]))
            .expect("append");
        let staged = log.paths().data_dir().join("p-00000000000000ff.zyr.tmp");
        fs::write(&staged, b"half written").expect("stage");

        let report = cleanup_orphans(&log, log.latest_version()).expect("cleanup");
        assert_eq!(report.staged_files, 1);
        assert!(
            staged.exists(),
            "an in-flight write may be producing this right now"
        );
    }
}

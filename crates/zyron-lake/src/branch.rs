//! Branches: alternate log heads over one table.
//!
//! A branch shares main's history up to its fork point and keeps its own
//! version files after it, so creating one writes a marker and copies no
//! data whatever the table's size. Branch writes land in the shared data
//! directory under partition ids nobody else can pick, which is what lets a
//! merge be pure metadata: main starts referencing files that already exist.
//!
//! Merging is a three-way file-set merge against the fork point. A file both
//! sides rewrote is reported, never resolved, because the two rewrites are
//! different row images of the same source rows and picking one silently
//! loses the other.

use std::collections::BTreeMap;
use std::fs;

use zyron_common::ZyronError;

use crate::manifest::{DeletePredicate, ManifestFile, PartitionEntry};
use crate::paths::LakePaths;
use crate::transaction_log::{CommitAttempt, LogEntry, TransactionLog};

/// Marker file identifying a branch and its fork point.
const BRANCH_MARKER: &str = "_base";
const MARKER_MAGIC: [u8; 4] = *b"ZYBR";
const MARKER_LEN: usize = 24;

/// One branch of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchInfo {
    pub name: String,
    /// Main version the branch forked from
    pub base_version: u64,
    pub created_us: i64,
    /// Newest version on the branch, equal to `base_version` until it is
    /// written to
    pub head_version: u64,
}

/// What a merge applied to main.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeOutcome {
    /// The version the merge committed, None when the branch had nothing
    /// main did not already have
    pub version: Option<u64>,
    pub files_added: usize,
    pub files_removed: usize,
    pub predicates_added: usize,
}

fn encode_marker(base_version: u64, created_us: i64) -> [u8; MARKER_LEN] {
    let mut buf = [0u8; MARKER_LEN];
    buf[0..4].copy_from_slice(&MARKER_MAGIC);
    buf[4..12].copy_from_slice(&base_version.to_le_bytes());
    buf[12..20].copy_from_slice(&created_us.to_le_bytes());
    let crc = zyron_common::hash32(&buf[0..20]);
    buf[20..24].copy_from_slice(&crc.to_le_bytes());
    buf
}

fn decode_marker(bytes: &[u8], ctx: &str) -> Result<(u64, i64), ZyronError> {
    if bytes.len() != MARKER_LEN || bytes[0..4] != MARKER_MAGIC {
        return Err(ZyronError::ManifestCorrupted {
            path: ctx.to_string(),
            reason: "branch marker is not a branch marker".into(),
        });
    }
    let stored = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    if stored != zyron_common::hash32(&bytes[0..20]) {
        return Err(ZyronError::ManifestCorrupted {
            path: ctx.to_string(),
            reason: "branch marker checksum mismatch".into(),
        });
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[4..12]);
    let base_version = u64::from_le_bytes(b);
    b.copy_from_slice(&bytes[12..20]);
    let created_us = i64::from_le_bytes(b);
    Ok((base_version, created_us))
}

fn marker_path(paths: &LakePaths, name: &str) -> std::path::PathBuf {
    paths.branch_dir(name).join(BRANCH_MARKER)
}

/// Rejects a name that cannot be a directory component, so a branch can
/// never address anything outside its table's log.
fn validate_name(name: &str) -> Result<(), ZyronError> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if ok {
        Ok(())
    } else {
        Err(ZyronError::BranchConflict(format!(
            "branch name \"{}\" must be 1 to 128 characters of letters, digits, underscore or dash",
            name
        )))
    }
}

/// Creates a branch at a main version, defaulting to main's head.
///
/// Writes one marker file. No data file is read, written or copied, so the
/// cost is the same for an empty table and a petabyte one.
pub fn create_branch(
    main: &TransactionLog,
    name: &str,
    from_version: Option<u64>,
    created_us: i64,
) -> Result<BranchInfo, ZyronError> {
    validate_name(name)?;
    if main.branch_name().is_some() {
        return Err(ZyronError::BranchConflict(
            "a branch is created from the table's main log".into(),
        ));
    }
    let head = main.latest_version();
    let base_version = from_version.unwrap_or(head);
    if base_version == 0 || base_version > head {
        return Err(ZyronError::VersionNotFound(base_version));
    }
    // The fork point must be readable, otherwise the branch is born broken
    main.manifest_at(base_version)?;

    let dir = main.paths().branch_dir(name);
    fs::create_dir_all(&dir)?;
    let path = marker_path(main.paths(), name);
    let mut file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(ZyronError::BranchAlreadyExists(name.to_string()));
        }
        Err(e) => return Err(e.into()),
    };
    use std::io::Write;
    file.write_all(&encode_marker(base_version, created_us))?;
    file.sync_all()?;

    Ok(BranchInfo {
        name: name.to_string(),
        base_version,
        created_us,
        head_version: base_version,
    })
}

/// Reads a branch's marker.
pub fn branch_info(paths: &LakePaths, name: &str) -> Result<BranchInfo, ZyronError> {
    validate_name(name)?;
    let path = marker_path(paths, name);
    let bytes = fs::read(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => ZyronError::BranchNotFound(name.to_string()),
        _ => ZyronError::IoError(format!(
            "cannot read branch marker {}: {}",
            path.display(),
            e
        )),
    })?;
    let (base_version, created_us) = decode_marker(&bytes, &path.to_string_lossy())?;
    let head_version = head_on_disk(paths, name, base_version)?;
    Ok(BranchInfo {
        name: name.to_string(),
        base_version,
        created_us,
        head_version,
    })
}

/// The newest contiguous branch version present, without opening the log.
fn head_on_disk(paths: &LakePaths, name: &str, base_version: u64) -> Result<u64, ZyronError> {
    let dir = paths.branch_dir(name);
    let mut versions = Vec::new();
    for dirent in fs::read_dir(&dir)? {
        let dirent = dirent?;
        let file_name = dirent.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if let Some((v, crate::paths::VersionFileKind::Version)) =
            crate::paths::parse_version_file_name(file_name)
        {
            if v > base_version {
                versions.push(v);
            }
        }
    }
    versions.sort_unstable();
    let mut head = base_version;
    for v in versions {
        if v != head + 1 {
            break;
        }
        head = v;
    }
    Ok(head)
}

/// Every branch of a table, by name.
pub fn list_branches(paths: &LakePaths) -> Result<Vec<BranchInfo>, ZyronError> {
    let root = paths.log_dir().join("branches");
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for dirent in fs::read_dir(&root)? {
        let dirent = dirent?;
        if !dirent.file_type()?.is_dir() {
            continue;
        }
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        // A directory without a readable marker is not a branch, skipping it
        // keeps one damaged branch from hiding the others
        if let Ok(info) = branch_info(paths, name) {
            out.push(info);
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Opens a branch as a writable head over main's shared history.
pub fn open_branch(paths: &LakePaths, name: &str) -> Result<TransactionLog, ZyronError> {
    let info = branch_info(paths, name)?;
    TransactionLog::open_branch(paths.clone(), name, info.base_version)
}

/// Opens a branch through the process-global registry, one shared head per
/// branch.
///
/// Writing needs this rather than [`open_branch`]: a commit refuses to build
/// on another transaction's pending version, and a pending version lives on
/// the instance that created it, so two instances of one branch would each
/// think the other's uncommitted version is already published.
pub fn open_branch_shared(
    paths: &LakePaths,
    name: &str,
) -> Result<std::sync::Arc<TransactionLog>, ZyronError> {
    validate_name(name)?;
    let key = paths.branch_dir(name);
    if let Some(hit) = TransactionLog::lookup_registered(&key) {
        return Ok(hit);
    }
    let info = branch_info(paths, name)?;
    let log = TransactionLog::open_branch(paths.clone(), name, info.base_version)?;
    TransactionLog::share_registered(key, log)
}

/// Deletes a branch's versions and marker.
///
/// Data files stay: they are referenced by partition id, and a file the
/// branch added that main never merged is unreachable from any manifest, so
/// the table's vacuum reclaims it with every other unreferenced file.
pub fn drop_branch(paths: &LakePaths, name: &str) -> Result<(), ZyronError> {
    validate_name(name)?;
    let dir = paths.branch_dir(name);
    if !dir.exists() {
        return Err(ZyronError::BranchNotFound(name.to_string()));
    }
    // The shared head goes with the directory, otherwise a branch recreated
    // under the same name would get the dropped branch's cached versions
    TransactionLog::remove_registered(&dir);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Sorted partition ids of a manifest's live files.
fn file_ids(manifest: &ManifestFile) -> Vec<u64> {
    manifest.entries.iter().map(|e| e.partition_id).collect()
}

/// Ids in `a` that are not in `b`, both sorted ascending.
fn difference(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut j = 0usize;
    for &id in a {
        while j < b.len() && b[j] < id {
            j += 1;
        }
        if j >= b.len() || b[j] != id {
            out.push(id);
        }
    }
    out
}

/// Ids present in both, both sorted ascending.
fn intersection(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut j = 0usize;
    for &id in a {
        while j < b.len() && b[j] < id {
            j += 1;
        }
        if j < b.len() && b[j] == id {
            out.push(id);
        }
    }
    out
}

/// Merges a branch into main.
///
/// The merge is three-way against the fork point: the branch's added files
/// and recorded predicates are replayed onto main's head, and its removals
/// are applied to files main still has. Nothing is rewritten, so a merge of
/// a terabyte branch is one small version file.
///
/// A file both sides removed is a conflict and is reported rather than
/// resolved: each side wrote its own replacement rows for that file, so
/// taking either one silently discards the other's edit.
pub fn merge_branch(
    main: &TransactionLog,
    name: &str,
    attempt: CommitAttempt<'_>,
) -> Result<MergeOutcome, ZyronError> {
    if main.branch_name().is_some() {
        return Err(ZyronError::BranchConflict(
            "a branch merges into the table's main log".into(),
        ));
    }
    let info = branch_info(main.paths(), name)?;
    // The shared head, so a version still pending under an open transaction
    // is below its published watermark and stays out of the merge
    let branch = open_branch_shared(main.paths(), name)?;
    if branch.latest_version() == info.base_version {
        return Ok(MergeOutcome {
            version: None,
            files_added: 0,
            files_removed: 0,
            predicates_added: 0,
        });
    }

    let base = main.manifest_at(info.base_version)?;
    let theirs = branch.latest_manifest()?;
    let base_ids = file_ids(&base);
    let their_ids = file_ids(&theirs);
    let branch_added = difference(&their_ids, &base_ids);
    let branch_removed = difference(&base_ids, &their_ids);

    // Predicates the branch recorded that the fork point did not have
    let branch_predicates: Vec<&DeletePredicate> = theirs
        .delete_predicates
        .iter()
        .filter(|p| p.created_version > info.base_version)
        .collect();

    let mut files_added = 0usize;
    let mut files_removed = 0usize;
    let mut predicates_added = 0usize;

    let version = main.commit(attempt, |ours| {
        let our_ids = file_ids(ours);
        let main_removed = difference(&base_ids, &our_ids);
        let both_removed = intersection(&branch_removed, &main_removed);
        if !both_removed.is_empty() {
            let listed: Vec<String> = both_removed
                .iter()
                .take(8)
                .map(|id| format!("{:016x}", id))
                .collect();
            return Err(ZyronError::BranchConflict(format!(
                "branch \"{}\" and main both rewrote {} file(s) since version {}: {}",
                name,
                both_removed.len(),
                info.base_version,
                listed.join(", ")
            )));
        }
        if theirs.schema.schema_id != base.schema.schema_id
            && ours.schema.schema_id != base.schema.schema_id
            && ours.schema.schema_id != theirs.schema.schema_id
        {
            return Err(ZyronError::BranchConflict(format!(
                "branch \"{}\" and main both changed the schema since version {}",
                name, info.base_version
            )));
        }

        let mut entries = Vec::new();
        // Removals first, then predicates, then adds. A predicate attaches
        // only to files already in the manifest, so the branch's own new
        // files never inherit a predicate the branch already applied to them
        for id in &branch_removed {
            if ours.entry_for(*id).is_some() {
                entries.push(LogEntry::RemoveFile { partition_id: *id });
            }
        }
        let mut next_id = ours
            .delete_predicates
            .iter()
            .map(|p| p.id)
            .chain(theirs.delete_predicates.iter().map(|p| p.id))
            .max()
            .unwrap_or(0)
            + 1;
        for predicate in &branch_predicates {
            entries.push(LogEntry::AddDeletePredicate(DeletePredicate {
                id: next_id,
                sql: predicate.sql.clone(),
                predicate: predicate.predicate.clone(),
                created_version: 0,
                // Carried from the branch that recorded it. The count is
                // what a compaction would reclaim, and re-counting it here
                // would read every file the merge touches
                pending_rows: predicate.pending_rows,
            }));
            next_id += 1;
        }
        for id in &branch_added {
            if let Some(entry) = theirs.entry_for(*id) {
                let mut entry: PartitionEntry = entry.clone();
                entry.added_version = 0;
                entries.push(LogEntry::AddFile(entry));
            }
        }
        if theirs.schema.schema_id > ours.schema.schema_id {
            entries.push(LogEntry::SchemaChange(theirs.schema.clone()));
        }
        for (key, value) in schema_property_diff(&base, &theirs) {
            entries.push(LogEntry::SetProperty { key, value });
        }

        files_removed = entries
            .iter()
            .filter(|e| matches!(e, LogEntry::RemoveFile { .. }))
            .count();
        files_added = branch_added.len();
        predicates_added = branch_predicates.len();
        Ok(entries)
    })?;

    Ok(MergeOutcome {
        version: Some(version),
        files_added,
        files_removed,
        predicates_added,
    })
}

/// Properties the branch set or changed since the fork point.
fn schema_property_diff(base: &ManifestFile, theirs: &ManifestFile) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (key, value) in &theirs.properties {
        if base.properties.get(key) != Some(value) {
            out.insert(key.clone(), value.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{append_rows, delete_where};
    use crate::predicate::{CompareOp, LakePredicate, LakeValue};
    use crate::schema::{LakeColumn, LakeSchema};
    use crate::transaction_log::OperationKind;
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
            commit_lsn: 1,
            timestamp_us,
            read_predicate: None,
            read_version: 0,
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
            LakePaths::new(dir, 13),
            attempt(OperationKind::SchemaChange, 100),
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create")
    }

    fn live_rows(manifest: &ManifestFile) -> u64 {
        manifest.entries.iter().map(|e| e.row_count).sum()
    }

    #[test]
    fn test_creating_a_branch_copies_no_data() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            13,
            &rows(&[1, 2, 3]),
        )
        .expect("append");

        let data_before: Vec<_> = fs::read_dir(log.paths().data_dir())
            .expect("data dir")
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect();

        let info = create_branch(&log, "staging", None, 999).expect("create branch");
        assert_eq!(info.base_version, 2);
        assert_eq!(info.head_version, 2, "a fresh branch has main's head");

        let data_after: Vec<_> = fs::read_dir(log.paths().data_dir())
            .expect("data dir")
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect();
        assert_eq!(data_before, data_after, "no data file was written");

        // The branch reads main's rows without any version of its own
        let branch = open_branch(log.paths(), "staging").expect("open");
        assert_eq!(branch.latest_version(), 2);
        assert_eq!(live_rows(&branch.latest_manifest().expect("manifest")), 3);

        assert_eq!(list_branches(log.paths()).expect("list").len(), 1);
        assert!(matches!(
            create_branch(&log, "staging", None, 1),
            Err(ZyronError::BranchAlreadyExists(_))
        ));
    }

    #[test]
    fn test_branch_writes_stay_off_main_until_merged() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            13,
            &rows(&[1, 2]),
        )
        .expect("append");
        create_branch(&log, "work", None, 300).expect("create branch");

        let branch = open_branch(log.paths(), "work").expect("open");
        append_rows(
            &branch,
            attempt(OperationKind::Append, 400),
            13,
            &rows(&[7, 8, 9]),
        )
        .expect("branch append");
        assert_eq!(branch.latest_version(), 3);
        assert_eq!(live_rows(&branch.latest_manifest().expect("manifest")), 5);

        // Main is untouched, and the branch version is not in main's log
        assert_eq!(log.latest_version(), 2);
        assert_eq!(live_rows(&log.latest_manifest().expect("manifest")), 2);
        assert!(!log.paths().version_file(3).exists());
        assert!(log.paths().branch_version_file("work", 3).exists());

        let outcome =
            merge_branch(&log, "work", attempt(OperationKind::Merge, 500)).expect("merge");
        assert_eq!(outcome.files_added, 1);
        assert_eq!(outcome.files_removed, 0);
        assert_eq!(outcome.version, Some(3));
        assert_eq!(live_rows(&log.latest_manifest().expect("manifest")), 5);

        // Merging again after the branch is dropped is not possible
        drop_branch(log.paths(), "work").expect("drop");
        assert!(matches!(
            open_branch(log.paths(), "work"),
            Err(ZyronError::BranchNotFound(_))
        ));
        assert!(list_branches(log.paths()).expect("list").is_empty());
    }

    #[test]
    fn test_merge_carries_a_branch_delete_without_rewriting_files() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            13,
            &rows(&[1, 2, 3, 40, 50]),
        )
        .expect("append");
        create_branch(&log, "prune", None, 300).expect("create branch");

        let branch = open_branch(log.paths(), "prune").expect("open");
        delete_where(
            &branch,
            attempt(OperationKind::Delete, 400),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(10),
            },
            "id < 10",
        )
        .expect("branch delete");

        // Main still sees every row until the merge lands
        assert_eq!(live_rows(&log.latest_manifest().expect("manifest")), 5);

        let outcome =
            merge_branch(&log, "prune", attempt(OperationKind::Merge, 500)).expect("merge");
        assert_eq!(
            outcome.predicates_added, 1,
            "the recorded predicate carries over"
        );
        let merged = log.latest_manifest().expect("manifest");
        assert_eq!(merged.delete_predicates.len(), 1);
        assert_eq!(
            merged.entries[0].delete_predicate_ids.len(),
            1,
            "the predicate attached to the file main already had"
        );
    }

    #[test]
    fn test_merge_branch_conflicting_file_is_reported_not_silently_resolved() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            13,
            &rows(&[1, 2, 3]),
        )
        .expect("append");
        create_branch(&log, "diverge", None, 300).expect("create branch");

        // Both sides replace the same base file with their own rewrite
        let branch = open_branch(log.paths(), "diverge").expect("open");
        let predicate = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Lt,
            value: LakeValue::Int(100),
        };
        delete_where(
            &branch,
            attempt(OperationKind::Delete, 400),
            &predicate,
            "id < 100",
        )
        .expect("branch delete");
        delete_where(
            &log,
            attempt(OperationKind::Delete, 450),
            &predicate,
            "id < 100",
        )
        .expect("main delete");

        let err = merge_branch(&log, "diverge", attempt(OperationKind::Merge, 500))
            .expect_err("both sides rewrote the same file");
        match err {
            ZyronError::BranchConflict(message) => {
                assert!(message.contains("diverge"), "{message}");
                assert!(message.contains("both rewrote"), "{message}");
            }
            other => panic!("expected a branch conflict, got {other:?}"),
        }

        // Main is untouched by the refused merge
        assert_eq!(log.latest_version(), 3);
    }

    #[test]
    fn test_a_branch_with_no_writes_merges_to_nothing() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(OperationKind::Append, 200), 13, &rows(&[1])).expect("append");
        create_branch(&log, "idle", None, 300).expect("create branch");

        let outcome =
            merge_branch(&log, "idle", attempt(OperationKind::Merge, 400)).expect("merge");
        assert_eq!(outcome.version, None);
        assert_eq!(log.latest_version(), 2, "nothing was committed");
    }

    #[test]
    fn test_branch_from_a_past_version_sees_that_version() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(OperationKind::Append, 200),
            13,
            &rows(&[1, 2]),
        )
        .expect("append one");
        append_rows(
            &log,
            attempt(OperationKind::Append, 300),
            13,
            &rows(&[3, 4, 5]),
        )
        .expect("append two");

        create_branch(&log, "old", Some(2), 400).expect("create branch");
        let branch = open_branch(log.paths(), "old").expect("open");
        assert_eq!(branch.branch_base(), 2);
        assert_eq!(
            live_rows(&branch.latest_manifest().expect("manifest")),
            2,
            "the branch sees the table as it was at version two"
        );

        assert!(matches!(
            create_branch(&log, "future", Some(99), 400),
            Err(ZyronError::VersionNotFound(99))
        ));
    }

    #[test]
    fn test_branch_names_are_restricted_to_directory_safe_text() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        for bad in ["", "../escape", "a/b", "with space", "dot.dot"] {
            assert!(
                matches!(
                    create_branch(&log, bad, None, 1),
                    Err(ZyronError::BranchConflict(_))
                ),
                "name {bad:?} must be refused"
            );
        }
        assert!(create_branch(&log, "ok-name_1", None, 1).is_ok());
    }
}

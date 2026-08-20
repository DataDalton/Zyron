//! `CREATE TABLE b CLONE OF a`, without copying a byte of data.
//!
//! A clone starts life holding exactly the file set the source held at the
//! version it was taken from. Data files are immutable and addressed by
//! partition id, so two tables can hold the same file at the same time and
//! neither can change what the other sees. That immutability is what makes
//! the whole thing safe, and it is a property of the format rather than
//! something this module arranges.
//!
//! ## How the files are shared
//!
//! Each file is **hard linked** into the clone's data directory. The bytes
//! are not copied: both directory entries name the same data, and the
//! filesystem keeps it alive until the last entry is gone. That is what
//! makes a clone of a terabyte table finish in the time it takes to walk
//! the manifest.
//!
//! The alternative was to point the clone's paths at the source's data
//! directory, the way a follower reads a leader's files. That mechanism
//! exists (`LakePaths::with_shared_data`) and it was the wrong fit here:
//! the redirect has to be known everywhere a table's paths are derived,
//! which is eighty-nine places across the engine, and every one of them
//! would have had to learn to ask whether this table is a clone. A hard
//! link puts the answer in the filesystem, where every one of those places
//! already looks.
//!
//! ## The pin
//!
//! The clone also writes a pin under the source's log,
//! `_refs/<clone_table_id>.ref`, naming the version it was taken at. The
//! source's vacuum reads those and treats the pinned versions' files as
//! reachable.
//!
//! The pin is not what keeps the clone's data alive, since the hard link
//! does that. It keeps the *source* able to serve the version the clone was
//! taken from, which is what a later repair, a re-clone, or a reader asking
//! the source for that version all need. Without it, vacuuming the source
//! would leave the clone holding files the source no longer admits to
//! having, and the relationship would be invisible to everything except an
//! inode count.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use zyron_common::ZyronError;

use crate::paths::LakePaths;
use crate::transaction_log::{CommitAttempt, LogEntry, TransactionLog};

/// Table this one was cloned from, as its table id
pub const CLONE_SOURCE_TABLE_PROPERTY: &str = "clone_source_table_id";

/// Version of that table this one was cloned at
pub const CLONE_SOURCE_VERSION_PROPERTY: &str = "clone_source_version";

/// What a clone took, for the statement to report
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloneOutcome {
    /// Version of the source the clone was taken from
    pub source_version: u64,
    pub files: usize,
    pub index_files: usize,
    pub rows: u64,
    /// Bytes the clone now holds and did not copy
    pub bytes_shared: u64,
}

/// Creates `clone_paths` as a copy of `source` at `at_version`, or at the
/// source's head when that is None.
///
/// Every file the source held at that version is hard linked into the
/// clone's data directory and named in the clone's first version, so the
/// clone is readable the moment this returns and shares storage with the
/// source until one of them rewrites it.
pub fn clone_table(
    source: &TransactionLog,
    clone_paths: LakePaths,
    attempt: CommitAttempt<'_>,
    at_version: Option<u64>,
) -> Result<(TransactionLog, CloneOutcome), ZyronError> {
    let head = source.head_version();
    let version = at_version.unwrap_or(head);
    if version == 0 || version > head {
        return Err(ZyronError::Internal(format!(
            "cannot clone version {version}, the source has versions 1 through {head}"
        )));
    }
    let manifest = source.manifest_at(version)?;

    let clone_id = clone_paths.table_id().ok_or_else(|| {
        ZyronError::Internal("a clone needs a table id in its root directory name".into())
    })?;

    fs::create_dir_all(clone_paths.data_dir())?;

    // Links first. A link that fails leaves the clone's log uncreated, so
    // the statement fails with nothing half made, and the links that did
    // land are cleaned up on the way out
    let mut linked: Vec<std::path::PathBuf> = Vec::new();
    let mut link_all = || -> Result<(), ZyronError> {
        for entry in &manifest.entries {
            let from = source.paths().data_file(entry.partition_id);
            let to = clone_paths.data_file(entry.partition_id);
            link_file(&from, &to)?;
            linked.push(to);
        }
        for file in &manifest.index_files {
            let from = source
                .paths()
                .index_file(file.index_id, file.file.partition_id);
            let to = clone_paths.index_file(file.index_id, file.file.partition_id);
            link_file(&from, &to)?;
            linked.push(to);
        }
        Ok(())
    };
    if let Err(e) = link_all() {
        for path in &linked {
            let _ = fs::remove_file(path);
        }
        return Err(e);
    }

    let mut entries = vec![LogEntry::SchemaChange(manifest.schema.clone())];
    if !manifest.cluster_spec.keys.is_empty() {
        entries.push(LogEntry::SetClusterSpec(manifest.cluster_spec.clone()));
    }
    for (key, value) in &manifest.properties {
        entries.push(LogEntry::SetProperty {
            key: key.clone(),
            value: value.clone(),
        });
    }
    // Where this came from, so the relationship survives a restart. A drop
    // reads it to release the pin, and without it the source would keep a
    // claim from a table that no longer exists
    let source_id = source.paths().table_id().ok_or_else(|| {
        ZyronError::Internal("a clone source needs a table id in its root directory name".into())
    })?;
    entries.push(LogEntry::SetProperty {
        key: CLONE_SOURCE_TABLE_PROPERTY.to_string(),
        value: source_id.to_string(),
    });
    entries.push(LogEntry::SetProperty {
        key: CLONE_SOURCE_VERSION_PROPERTY.to_string(),
        value: version.to_string(),
    });
    for index in &manifest.indexes {
        entries.push(LogEntry::AddIndex(index.clone()));
    }
    for entry in &manifest.entries {
        entries.push(LogEntry::AddFile(entry.clone()));
    }
    for file in &manifest.index_files {
        entries.push(LogEntry::AddIndexFile(file.clone()));
    }
    // Delete predicates travel too. A row the source had logically deleted
    // is deleted in the clone as well, because the clone is a copy of what
    // the source showed at that version, not of what its files hold
    for predicate in &manifest.delete_predicates {
        entries.push(LogEntry::AddDeletePredicate(predicate.clone()));
    }

    let log = match TransactionLog::create_from_entries(clone_paths, attempt, entries) {
        Ok(log) => log,
        Err(e) => {
            for path in &linked {
                let _ = fs::remove_file(path);
            }
            return Err(e);
        }
    };

    write_pin(source.paths(), clone_id, version)?;

    let outcome = CloneOutcome {
        source_version: version,
        files: manifest.entries.len(),
        index_files: manifest.index_files.len(),
        rows: manifest.entries.iter().map(|e| e.row_count).sum(),
        bytes_shared: manifest.entries.iter().map(|e| e.size_bytes).sum(),
    };
    Ok((log, outcome))
}

/// Links one file, tolerating a link that is already there.
///
/// A retried clone of the same source at the same version produces the
/// same link, and failing on that would make the retry the thing that
/// breaks
fn link_file(from: &Path, to: &Path) -> Result<(), ZyronError> {
    match fs::hard_link(from, to) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(ZyronError::Internal(format!(
            "cannot share {} with a clone: {e}. A clone links the source's files rather than \
             copying them, which needs both tables on one filesystem",
            from.display()
        ))),
    }
}

/// Records that a table is holding a version of this one.
///
/// The file's whole content is the version, so the source's vacuum can
/// decide what a clone still needs without opening the clone
fn write_pin(source: &LakePaths, clone_table_id: u32, version: u64) -> Result<(), ZyronError> {
    let path = source.clone_ref(clone_table_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, version.to_string().as_bytes())?;
    Ok(())
}

/// The table and version a clone was taken from, when this table is one.
///
/// Read from the clone's own manifest, so it survives a restart and is
/// visible to anything that can open the table
pub fn clone_source(manifest: &crate::manifest::ManifestFile) -> Option<(u32, u64)> {
    let table = manifest
        .properties
        .get(CLONE_SOURCE_TABLE_PROPERTY)?
        .trim()
        .parse::<u32>()
        .ok()?;
    let version = manifest
        .properties
        .get(CLONE_SOURCE_VERSION_PROPERTY)?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some((table, version))
}

/// Drops a table's pin on this one, which is what a DROP TABLE on the
/// clone does. Missing is success: the pin is a claim, and a claim nobody
/// is making is the state this is trying to reach
pub fn release_pin(source: &LakePaths, clone_table_id: u32) -> Result<(), ZyronError> {
    match fs::remove_file(source.clone_ref(clone_table_id)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Versions of this table that clones are still holding, by the table id
/// holding each.
///
/// An unreadable pin is reported as None against its holder rather than
/// skipped, because vacuum has to tell "no clone needs this" apart from "a
/// clone needs something and I could not find out what"
pub fn clone_pins(source: &LakePaths) -> BTreeMap<u32, Option<u64>> {
    let mut pins = BTreeMap::new();
    let dir = source.clone_ref(0);
    let Some(dir) = dir.parent().map(|p| p.to_path_buf()) else {
        return pins;
    };
    let Ok(read) = fs::read_dir(&dir) else {
        return pins;
    };
    for dirent in read.flatten() {
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(id) = name
            .strip_suffix(".ref")
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        let version = fs::read_to_string(dirent.path())
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok());
        pins.insert(id, version);
    }
    pins
}

/// Files a clone is still holding, and whether every pin could be read.
///
/// The flag is what vacuum needs: a pin whose version cannot be
/// reconstructed means the reachable set is unknown, and deleting on an
/// unknown reachable set is how a clone loses data
pub fn pinned_partitions(source: &TransactionLog) -> (std::collections::BTreeSet<u64>, bool) {
    let mut reachable = std::collections::BTreeSet::new();
    let mut complete = true;
    for (_, version) in clone_pins(source.paths()) {
        let Some(version) = version else {
            complete = false;
            continue;
        };
        match source.manifest_at(version) {
            Ok(manifest) => {
                for entry in &manifest.entries {
                    reachable.insert(entry.partition_id);
                }
            }
            Err(_) => complete = false,
        }
    }
    (reachable, complete)
}

//! AS OF resolution over the transaction log.
//!
//! A version query resolves directly through the log. A timestamp query
//! walks published versions newest first and returns the newest one
//! committed at or before the target instant. Commit timestamps come from
//! caller clocks and are not guaranteed monotone, which is why the walk
//! is a scan rather than a binary search, each probe is one 128-byte
//! header read so the cost is bounded by the retention window.
//!
//! Versions the log can no longer replay, removed by version GC, resolve
//! through the surviving checkpoints, and a target before everything
//! still reachable is an error rather than a silent nearest match

use std::fs;
use std::io::Read;

use zyron_common::ZyronError;

use crate::manifest::{MANIFEST_MAGIC, ManifestFile};
use crate::paths::{VersionFileKind, parse_version_file_name};
use crate::transaction_log::{TransactionLog, read_commit_header};

/// What an AS OF clause resolved to at the syntax layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeTravelSpec {
    /// VERSION AS OF n
    Version(u64),
    /// TIMESTAMP AS OF t, microseconds since the epoch
    Timestamp(i64),
}

/// Resolves a spec to a concrete published version
pub fn resolve_version(log: &TransactionLog, spec: TimeTravelSpec) -> Result<u64, ZyronError> {
    match spec {
        TimeTravelSpec::Version(v) => {
            if v == 0 || v > log.latest_version() {
                return Err(ZyronError::Internal(format!(
                    "version {} does not exist, table is at version {}",
                    v,
                    log.latest_version()
                )));
            }
            Ok(v)
        }
        TimeTravelSpec::Timestamp(target_us) => resolve_timestamp(log, target_us),
    }
}

/// Resolves a spec straight to its manifest
pub fn manifest_as_of(
    log: &TransactionLog,
    spec: TimeTravelSpec,
) -> Result<std::sync::Arc<ManifestFile>, ZyronError> {
    let version = resolve_version(log, spec)?;
    log.manifest_at(version)
}

fn resolve_timestamp(log: &TransactionLog, target_us: i64) -> Result<u64, ZyronError> {
    let published = log.latest_version();
    // Every version still on disk with its commit timestamp, version
    // files by header read, checkpoints by their manifest header
    let mut candidates: Vec<(u64, i64)> = Vec::new();
    for dirent in fs::read_dir(log.paths().log_dir())? {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        match parse_version_file_name(name) {
            Some((v, VersionFileKind::Version)) if v <= published => {
                let header = read_commit_header(&log.paths().version_file(v))?;
                candidates.push((v, header.timestamp_us));
            }
            Some((v, VersionFileKind::Checkpoint)) if v <= published => {
                let ts = read_checkpoint_timestamp(log, v)?;
                candidates.push((v, ts));
            }
            _ => {}
        }
    }
    candidates.sort_unstable_by_key(|(v, _)| *v);
    candidates.dedup_by_key(|(v, _)| *v);
    // Newest version committed at or before the target that the log can
    // still replay
    for (v, ts) in candidates.iter().rev() {
        if *ts <= target_us && log.manifest_at(*v).is_ok() {
            return Ok(*v);
        }
    }
    Err(ZyronError::Internal(format!(
        "no reachable version at or before timestamp {} us, the earliest retained commit is newer or retention removed it",
        target_us
    )))
}

/// Reads only the timestamp field out of a checkpoint's 64-byte header.
/// Integrity of the whole file is verified when the manifest is opened,
/// this read only has to be internally consistent
fn read_checkpoint_timestamp(log: &TransactionLog, version: u64) -> Result<i64, ZyronError> {
    let path = log.paths().checkpoint_file(version);
    let mut file = fs::File::open(&path)?;
    let mut head = [0u8; 64];
    file.read_exact(&mut head)?;
    if head[..4] != MANIFEST_MAGIC {
        return Err(ZyronError::ManifestCorrupted {
            path: path.to_string_lossy().to_string(),
            reason: "bad manifest magic".into(),
        });
    }
    let mut ts = [0u8; 8];
    ts.copy_from_slice(&head[24..32]);
    Ok(i64::from_le_bytes(ts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::PartitionEntry;
    use crate::paths::LakePaths;
    use crate::schema::{LakeColumn, LakeSchema};
    use crate::transaction_log::{CommitAttempt, LogEntry, OperationKind};
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

    fn attempt_at(ts: i64) -> CommitAttempt<'static> {
        CommitAttempt {
            operation: OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 1,
            timestamp_us: ts,
            read_predicate: None,
            read_version: 0,
            audit: None,
        }
    }

    fn bare_file(partition_id: u64) -> PartitionEntry {
        PartitionEntry {
            partition_id,
            size_bytes: 64,
            row_count: 1,
            added_version: 0,
            cluster_spec_id: 0,
            column_stats: std::sync::Arc::new(vec![]),
            delete_predicate_ids: vec![],
        }
    }

    fn build_log(dir: &std::path::Path) -> TransactionLog {
        let mut create = attempt_at(1_000);
        create.operation = OperationKind::SchemaChange;
        let log = TransactionLog::create(
            LakePaths::new(dir, 7),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create");
        // Versions 2, 3, 4 at timestamps 2000, 3000, 4000
        for (i, ts) in [(0u64, 2_000i64), (1, 3_000), (2, 4_000)] {
            log.commit(attempt_at(ts), |_| {
                Ok(vec![LogEntry::AddFile(bare_file(0x100 + i))])
            })
            .expect("append");
        }
        log
    }

    #[test]
    fn test_version_resolution_bounds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = build_log(dir.path());
        assert_eq!(
            resolve_version(&log, TimeTravelSpec::Version(3)).expect("resolves"),
            3
        );
        assert!(resolve_version(&log, TimeTravelSpec::Version(0)).is_err());
        assert!(resolve_version(&log, TimeTravelSpec::Version(9)).is_err());
    }

    #[test]
    fn test_timestamp_resolution() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = build_log(dir.path());
        // Between versions 2 and 3 lands on 2
        assert_eq!(
            resolve_version(&log, TimeTravelSpec::Timestamp(2_500)).expect("resolves"),
            2
        );
        // Exactly at a commit includes it
        assert_eq!(
            resolve_version(&log, TimeTravelSpec::Timestamp(3_000)).expect("resolves"),
            3
        );
        // After the head lands on the head
        assert_eq!(
            resolve_version(&log, TimeTravelSpec::Timestamp(9_999)).expect("resolves"),
            4
        );
        // Before the table existed is an error
        assert!(resolve_version(&log, TimeTravelSpec::Timestamp(500)).is_err());
        // The resolved manifest reflects that version's state
        let m = manifest_as_of(&log, TimeTravelSpec::Timestamp(3_500)).expect("manifest");
        assert_eq!(m.snapshot_id, 3);
        assert_eq!(m.entries.len(), 2);
    }

    #[test]
    fn test_timestamp_resolution_survives_gc_via_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = build_log(dir.path());
        log.checkpoint(3).expect("checkpoint");
        let removed = log.gc_versions(3).expect("gc");
        assert!(removed > 0);
        // Version 2's file is gone, the nearest reachable state at that
        // instant is the version 3 checkpoint, which is newer than the
        // target, so resolution refuses rather than answering wrong
        assert!(resolve_version(&log, TimeTravelSpec::Timestamp(2_500)).is_err());
        // The checkpoint version itself resolves by its manifest timestamp
        assert_eq!(
            resolve_version(&log, TimeTravelSpec::Timestamp(3_500)).expect("resolves"),
            3
        );
        assert_eq!(
            resolve_version(&log, TimeTravelSpec::Timestamp(9_999)).expect("resolves"),
            4
        );
    }
}

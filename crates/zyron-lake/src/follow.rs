//! Following another node's transaction log.
//!
//! The log is the replication protocol. A version file is a complete,
//! self-describing description of one commit: which files entered the
//! table, which left, which predicates were recorded. A follower that
//! applies those entries in order arrives at exactly the leader's state,
//! because with one writer per dataset the log is a total order and
//! applying it is a replay rather than a merge.
//!
//! **Ship the log, not the data.** On shared storage a data file the
//! leader wrote is a file the follower can already open, so a sync moves
//! version files, which are metadata, and copies no data at all. Where
//! storage is not shared the same entries name the files to fetch, and
//! each one transfers at most once because a `.zyr` is immutable.
//!
//! The follower's log is its own. It numbers versions the same way the
//! leader does, so a follower that has applied through version N holds
//! byte-identical manifests to the leader's version N, and freshness is
//! the gap between the two numbers rather than a timestamp comparison
//! across two clocks.

use std::fs;
use std::io::Write;
use std::path::Path;

use zyron_common::ZyronError;

use crate::paths::{LakePaths, VersionFileKind, parse_version_file_name};
use crate::transaction_log::{
    CommitAttempt, LogEntry, OperationKind, TransactionLog, VersionFileData,
};

/// File name of the follower's cursor under its log directory.
const CURSOR_FILE: &str = "_followed";

/// One of the leader's versions, ready to apply.
#[derive(Debug, Clone)]
pub struct FollowedVersion {
    pub version: u64,
    pub timestamp_us: i64,
    pub operation: OperationKind,
    pub entries: Vec<LogEntry>,
}

/// How far behind a follower is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Freshness {
    /// Newest version the leader has published
    pub leader_version: u64,
    /// Newest version the follower has applied
    pub follower_version: u64,
    /// Microseconds since the epoch of the newest applied version, zero
    /// when nothing has been applied
    pub applied_us: i64,
}

impl Freshness {
    /// Versions the follower has not applied yet.
    ///
    /// A count rather than a duration on purpose: two nodes do not share a
    /// clock, and a follower that is one version behind a table nobody
    /// writes is current, not stale
    pub fn lag_versions(&self) -> u64 {
        self.leader_version.saturating_sub(self.follower_version)
    }

    pub fn is_current(&self) -> bool {
        self.follower_version >= self.leader_version
    }
}

/// Reads the leader's versions after `from`, in order.
///
/// Reads version files directly rather than opening the leader's log,
/// because a follower must not register itself as a second holder of a
/// table it does not write. `limit` bounds one poll so a follower that
/// fell far behind catches up in bounded steps rather than one long stall.
pub fn read_versions_after(
    leader: &LakePaths,
    from: u64,
    limit: usize,
) -> Result<Vec<FollowedVersion>, ZyronError> {
    let mut versions = Vec::new();
    let listing = match fs::read_dir(leader.log_dir()) {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(versions),
        Err(e) => return Err(e.into()),
    };
    let mut available: Vec<u64> = Vec::new();
    for dirent in listing {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some((version, VersionFileKind::Version)) = parse_version_file_name(name) {
            if version > from {
                available.push(version);
            }
        }
    }
    available.sort_unstable();
    // A gap means a version file is missing, and applying past it would
    // silently skip a commit. Stop at the gap and report what is
    // contiguous, so the follower stays behind rather than wrong
    let mut expected = from + 1;
    for version in available.into_iter().take(limit) {
        if version != expected {
            break;
        }
        let path = leader.version_file(version);
        let bytes = fs::read(&path)?;
        let data = VersionFileData::decode(&bytes, &path.to_string_lossy())?;
        versions.push(FollowedVersion {
            version,
            timestamp_us: data.header.timestamp_us,
            operation: data.header.operation,
            entries: data.entries,
        });
        expected += 1;
    }
    Ok(versions)
}

/// Applies the leader's versions to a follower's log.
///
/// Each one becomes a commit carrying the leader's own entries, so the
/// follower's manifest at version N matches the leader's version N. The
/// entries name data files by partition id, and on shared storage those
/// files are already readable, so nothing is copied here.
///
/// Returns the versions applied. Stops at the first version the follower
/// cannot apply and reports the error, because skipping one would leave
/// the follower's state neither the leader's nor its own.
pub fn apply_versions(
    follower: &TransactionLog,
    versions: &[FollowedVersion],
) -> Result<u64, ZyronError> {
    let mut applied = 0u64;
    // A replay originates nothing, so it must not claim ownership either.
    // Claiming here would make the follower the owner of the first version
    // it applied and the leader a foreign writer from the next one on,
    // which is the exact inversion of what a follower is
    follower.set_writer_identity(0);
    for version in versions {
        let expected = follower.latest_version() + 1;
        if version.version != expected {
            return Err(ZyronError::ConflictError {
                mine: format!("follower at version {}", follower.latest_version()),
                theirs: format!("leader version {}", version.version),
                reason: "a follower applies the leader's versions in order and numbers \
them the same, so a version out of sequence means a gap rather than a conflict"
                    .to_string(),
            });
        }
        let attempt = CommitAttempt {
            operation: version.operation,
            db_txn_id: 0,
            commit_lsn: 0,
            timestamp_us: version.timestamp_us,
            read_predicate: None,
            audit: None,
        };
        let entries = version.entries.clone();
        follower.commit(attempt, move |_| Ok(entries.clone()))?;
        applied += 1;
        // A replica writes as whoever owns the dataset, because the
        // versions it holds are that node's. Without this the first
        // version carrying the leader's ownership claim would make the
        // follower a foreign writer to its own log and every later version
        // would be refused. A local writer on this node is still refused,
        // which is correct: a follower is read-only to everyone but the
        // owner whose log it is replaying
        if let Ok(manifest) = follower.latest_manifest() {
            if let Some(owner) = crate::transaction_log::writer_node(&manifest) {
                follower.set_writer_identity(owner);
            }
        }
    }
    Ok(applied)
}

/// Reads and applies in one step, returning how far the follower now is.
///
/// The bytes this moves are version files. A data file the leader wrote is
/// never read, opened or copied here, which is what makes a sync on shared
/// storage metadata only.
pub fn sync(
    leader: &LakePaths,
    follower: &TransactionLog,
    limit: usize,
) -> Result<Freshness, ZyronError> {
    let from = follower.latest_version();
    let versions = read_versions_after(leader, from, limit)?;
    apply_versions(follower, &versions)?;
    let applied_us = versions.last().map(|v| v.timestamp_us).unwrap_or_else(|| {
        follower
            .latest_manifest()
            .map(|m| m.timestamp_us)
            .unwrap_or(0)
    });
    let freshness = Freshness {
        leader_version: leader_head(leader)?,
        follower_version: follower.latest_version(),
        applied_us,
    };
    save_cursor(follower.paths(), &freshness)?;
    Ok(freshness)
}

/// The leader's newest version, read from its log directory.
pub fn leader_head(leader: &LakePaths) -> Result<u64, ZyronError> {
    let listing = match fs::read_dir(leader.log_dir()) {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e.into()),
    };
    let mut head = 0u64;
    for dirent in listing {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some((version, VersionFileKind::Version)) = parse_version_file_name(name) {
            head = head.max(version);
        }
    }
    Ok(head)
}

/// Records how far a follower has caught up, so freshness survives a
/// restart and a reader can be told how stale its answer is without the
/// follower re-reading the leader's whole log first.
fn save_cursor(follower: &LakePaths, freshness: &Freshness) -> Result<(), ZyronError> {
    fs::create_dir_all(follower.log_dir())?;
    let path = follower.log_dir().join(CURSOR_FILE);
    let temp = path.with_extension("tmp");
    let body = format!(
        "{} {} {}\n",
        freshness.leader_version, freshness.follower_version, freshness.applied_us
    );
    {
        let mut file = fs::File::create(&temp)?;
        file.write_all(body.as_bytes())?;
        file.sync_all()?;
    }
    fs::rename(&temp, &path)?;
    Ok(())
}

/// The last recorded freshness, None when this table has never followed.
pub fn load_cursor(follower: &LakePaths) -> Option<Freshness> {
    let text = fs::read_to_string(follower.log_dir().join(CURSOR_FILE)).ok()?;
    let mut parts = text.split_whitespace();
    Some(Freshness {
        leader_version: parts.next()?.parse().ok()?,
        follower_version: parts.next()?.parse().ok()?,
        applied_us: parts.next()?.parse().ok()?,
    })
}

/// Bytes a directory's files occupy, for a caller proving a sync moved
/// none of them.
pub fn directory_bytes(dir: &Path) -> u64 {
    let Ok(listing) = fs::read_dir(dir) else {
        return 0;
    };
    listing
        .filter_map(|d| d.ok())
        .filter_map(|d| d.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Version files a follower would have to move to catch up, which is the
/// whole cost of a sync on shared storage.
pub fn pending_metadata_bytes(leader: &LakePaths, from: u64) -> u64 {
    let mut total = 0u64;
    let mut version = from + 1;
    while let Ok(meta) = fs::metadata(leader.version_file(version)) {
        total += meta.len();
        version += 1;
    }
    total
}

/// Version file data as the follower reads it, re-exported so a caller can
/// inspect a leader's commit without opening its log.
pub type LeaderVersion = VersionFileData;

/// Decodes lowercase hex back to bytes, None when the text is not hex.
///
/// A version file crosses the wire as hex because the text protocol has no
/// cell shape for an arbitrary byte. Rejecting malformed input here rather
/// than salvaging what parses keeps a truncated transfer from decoding as
/// a shorter, structurally valid commit.
pub fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(text.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

/// Turns a leader's published log rows into versions ready to apply.
///
/// The rows are  as the leader's log view returns
/// them. Each payload is a whole version file and is decoded and checked
/// the same way a follower reading the filesystem checks one, so a
/// corrupted transfer is refused here rather than applied.
///
/// Stops at the first version that is not the next one expected, which is
/// the same gap rule the filesystem reader follows.
pub fn decode_log_rows(
    from: u64,
    rows: &[(u64, String)],
) -> Result<Vec<FollowedVersion>, ZyronError> {
    let mut versions = Vec::with_capacity(rows.len());
    let mut expected = from + 1;
    for (version, payload) in rows {
        if *version != expected {
            break;
        }
        let bytes = decode_hex(payload).ok_or_else(|| ZyronError::ManifestCorrupted {
            path: format!("version {}", version),
            reason: "payload is not hex, so the transfer was truncated or mangled".to_string(),
        })?;
        let data = VersionFileData::decode(&bytes, &format!("version {}", version))?;
        versions.push(FollowedVersion {
            version: *version,
            timestamp_us: data.header.timestamp_us,
            operation: data.header.operation,
            entries: data.entries,
        });
        expected += 1;
    }
    Ok(versions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{append_rows, delete_where};
    use crate::predicate::{CompareOp, LakePredicate, LakeValue};
    use crate::schema::{LakeColumn, LakeSchema};
    use crate::transaction_log::AllCommitted;
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

    fn attempt() -> CommitAttempt<'static> {
        CommitAttempt {
            operation: OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 1,
            timestamp_us: 1_754_700_000_000_000,
            read_predicate: None,
            audit: None,
        }
    }

    fn batch(ids: &[i64]) -> Vec<ColumnData> {
        vec![ColumnData {
            column_id: 0,
            cells: ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
        }]
    }

    /// The claim the mesh rests on: syncing a table on shared storage moves
    /// version files and copies no data at all.
    #[test]
    fn test_shared_storage_sync_transfers_metadata_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let leader_paths = LakePaths::new(dir.path(), 1);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let leader = TransactionLog::create(
            leader_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("leader");
        append_rows(&leader, attempt(), 1, &batch(&[1, 2, 3])).expect("append");
        append_rows(&leader, attempt(), 1, &batch(&[100, 200])).expect("append");

        // The follower keeps its own log and reads the leader's data in
        // place, which is what shared storage means
        let follower_paths = LakePaths::new(dir.path(), 2).with_shared_data(&leader_paths);
        assert!(follower_paths.has_shared_data());
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let follower = TransactionLog::create(
            follower_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("follower");

        let data_before = directory_bytes(&leader_paths.data_dir());
        assert!(data_before > 0, "the leader wrote real files");
        let own_data_before = directory_bytes(&follower_paths.root().join("data"));

        let freshness = sync(&leader_paths, &follower, 64).expect("sync");
        assert!(freshness.is_current(), "{freshness:?}");
        assert_eq!(freshness.lag_versions(), 0);
        assert_eq!(freshness.follower_version, leader.latest_version());

        // Not one byte of data moved: the leader's files are untouched and
        // the follower created none of its own
        assert_eq!(directory_bytes(&leader_paths.data_dir()), data_before);
        assert_eq!(
            directory_bytes(&follower_paths.root().join("data")),
            own_data_before,
            "a follower on shared storage writes no data files"
        );

        // And it reads the leader's rows through its own manifest
        let manifest = follower.latest_manifest().expect("manifest");
        let rows: u64 = manifest.entries.iter().map(|e| e.row_count).sum();
        assert_eq!(rows, 5);
        for entry in &manifest.entries {
            let reader = crate::reader::LakeFileReader::open(&follower_paths, entry.partition_id)
                .expect("the follower opens the leader's file in place");
            assert_eq!(reader.row_count() as u64, entry.row_count);
        }
    }

    /// A follower's manifest at version N is the leader's version N, which
    /// is what lets freshness be a version gap rather than a clock compare
    #[test]
    fn test_a_follower_reaches_the_leader_state_version_for_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let leader_paths = LakePaths::new(dir.path(), 3);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let leader = TransactionLog::create(
            leader_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("leader");
        append_rows(&leader, attempt(), 3, &batch(&[1, 2])).expect("append");
        append_rows(&leader, attempt(), 3, &batch(&[300, 400])).expect("append");
        delete_where(
            &leader,
            attempt(),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(50),
            },
            "id < 50",
        )
        .expect("delete");

        let follower_paths = LakePaths::new(dir.path(), 4).with_shared_data(&leader_paths);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let follower = TransactionLog::create(
            follower_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("follower");

        // One version at a time, so the catch-up path is exercised rather
        // than a single bulk apply
        loop {
            let freshness = sync(&leader_paths, &follower, 1).expect("sync");
            if freshness.is_current() {
                break;
            }
        }

        let leader_manifest = leader.latest_manifest().expect("leader manifest");
        let follower_manifest = follower.latest_manifest().expect("follower manifest");
        assert_eq!(follower_manifest.snapshot_id, leader_manifest.snapshot_id);
        assert_eq!(follower_manifest.entries, leader_manifest.entries);
        assert_eq!(
            follower_manifest.delete_predicates,
            leader_manifest.delete_predicates
        );

        // Freshness survives a restart without re-reading the leader's log
        let cursor = load_cursor(&follower_paths).expect("cursor");
        assert!(cursor.is_current());
        assert_eq!(cursor.follower_version, leader.latest_version());

        // A follower that has caught up and polls again moves nothing
        let idle = sync(&leader_paths, &follower, 64).expect("idle sync");
        assert_eq!(idle.lag_versions(), 0);
    }

    /// A missing version file is a gap, and applying past it would skip a
    /// commit silently. The follower stops at the gap and stays behind
    #[test]
    fn test_a_gap_in_the_leader_log_stops_the_follower_rather_than_skipping() {
        let dir = tempfile::tempdir().expect("tempdir");
        let leader_paths = LakePaths::new(dir.path(), 5);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let leader = TransactionLog::create(
            leader_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("leader");
        for id in 0..4i64 {
            append_rows(&leader, attempt(), 5, &batch(&[id])).expect("append");
        }
        assert_eq!(leader.latest_version(), 5);

        // Version 3 goes missing, as a partial transfer would leave it
        fs::remove_file(leader_paths.version_file(3)).expect("remove");

        let versions = read_versions_after(&leader_paths, 1, 64).expect("read");
        assert_eq!(
            versions.iter().map(|v| v.version).collect::<Vec<_>>(),
            vec![2],
            "reading stops at the gap rather than jumping it"
        );

        let follower_paths = LakePaths::new(dir.path(), 6).with_shared_data(&leader_paths);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let follower = TransactionLog::create(
            follower_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("follower");
        let freshness = sync(&leader_paths, &follower, 64).expect("sync");
        assert_eq!(freshness.follower_version, 2);
        assert!(!freshness.is_current(), "the follower knows it is behind");
        assert_eq!(freshness.lag_versions(), 3);
    }

    /// The single-writer rule must not stop a follower from replaying the
    /// owner's log, and must still stop this node from writing to it
    #[test]
    fn test_a_follower_replays_an_owned_log_without_becoming_a_second_writer() {
        const LEADER: u64 = 0x1111_1111_1111_1111;
        const FOLLOWER: u64 = 0x2222_2222_2222_2222;

        let dir = tempfile::tempdir().expect("tempdir");
        let leader_paths = LakePaths::new(dir.path(), 9);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let leader = TransactionLog::create(
            leader_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("leader");
        leader.set_writer_identity(LEADER);
        append_rows(&leader, attempt(), 9, &batch(&[1, 2])).expect("leader claims and writes");
        append_rows(&leader, attempt(), 9, &batch(&[3, 4])).expect("leader writes again");
        assert_eq!(
            crate::transaction_log::writer_node(&leader.latest_manifest().expect("m")),
            Some(LEADER)
        );

        let follower_paths = LakePaths::new(dir.path(), 10).with_shared_data(&leader_paths);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let follower = TransactionLog::create(
            follower_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("follower");
        follower.set_writer_identity(FOLLOWER);

        // Replay carries the leader's ownership claim, and every version
        // after it still applies
        let freshness = sync(&leader_paths, &follower, 64).expect("sync");
        assert!(freshness.is_current(), "{freshness:?}");
        assert_eq!(
            crate::transaction_log::writer_node(&follower.latest_manifest().expect("m")),
            Some(LEADER)
        );

        // This node still cannot write to a dataset it does not own
        follower.set_writer_identity(FOLLOWER);
        let refused = append_rows(&follower, attempt(), 10, &batch(&[9]))
            .expect_err("a follower is read-only to its own node");
        assert!(refused.to_string().contains("1111111111111111"));

        // And more of the leader's versions still replay afterwards
        append_rows(&leader, attempt(), 9, &batch(&[5, 6])).expect("leader writes");
        let freshness = sync(&leader_paths, &follower, 64).expect("second sync");
        assert!(freshness.is_current(), "{freshness:?}");
        let rows: u64 = follower
            .latest_manifest()
            .expect("m")
            .entries
            .iter()
            .map(|e| e.row_count)
            .sum();
        assert_eq!(rows, 6);
    }

    /// The cost of a sync is version files, not data files
    #[test]
    fn test_the_cost_of_a_sync_is_metadata() {
        let dir = tempfile::tempdir().expect("tempdir");
        let leader_paths = LakePaths::new(dir.path(), 7);
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        let leader = TransactionLog::create(
            leader_paths.clone(),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("leader");
        let many: Vec<i64> = (0..2000).collect();
        append_rows(&leader, attempt(), 7, &batch(&many)).expect("append");

        let data_bytes = directory_bytes(&leader_paths.data_dir());
        let metadata_bytes = pending_metadata_bytes(&leader_paths, 1);
        assert!(data_bytes > 0 && metadata_bytes > 0);
        assert!(
            metadata_bytes * 4 < data_bytes,
            "shipping the log must cost far less than shipping the data, \
             metadata {metadata_bytes} against data {data_bytes}"
        );
        let _ = AllCommitted;
    }
}

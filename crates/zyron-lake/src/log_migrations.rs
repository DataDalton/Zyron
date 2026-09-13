//! Moves a transaction log commit record between 1.0 and 1.1.
//!
//! A 1.1 record may add a data or index file under the tag that carries
//! the schema id the file was written under, may carry the shapes columns
//! had under earlier schema ids, and names in its header the database
//! transaction an intent commit belongs to. A 1.0 record holds none of
//! these, every entry it holds reads unchanged at 1.1, and the header
//! bytes that name the owner are zero in it, which reads as an intent
//! commit whose owner is not recorded. So moving one forward is a restamp
//! of the version the header names and of the checksum that covers the
//! header. Moving a 1.1 record back is the same restamp when the record
//! holds nothing a 1.0 reader lacks a tag for, and is refused otherwise
//! rather than dropping what the record says

use zyron_common::format::FormatKind;
use zyron_common::format::envelope;
use zyron_common::format::registry::FormatMigrator;
use zyron_common::format::version::FormatVersion;

use crate::format::{LAKE_LOG_FORMAT_VERSION, LAKE_LOG_FORMAT_VERSION_1_0};
use crate::transaction_log::{COMMIT_HEADER_LEN, LogEntry, VersionFileData};

inventory::submit! {
    FormatMigrator {
        kind: FormatKind::LakeTransactionLog,
        from: LAKE_LOG_FORMAT_VERSION_1_0,
        to: LAKE_LOG_FORMAT_VERSION,
        reversible: true,
        forward: log_1_0_to_1_1,
        backward: Some(log_1_1_to_1_0),
        no_body_change: false,
        description: "added files may carry the schema id they were written under and a record may carry column type history",
    }
}

/// Re-stamps a commit record at 1.1. The body is what it was, since every
/// 1.0 entry reads the same way at 1.1
pub fn log_1_0_to_1_1(file: &[u8]) -> Result<Vec<u8>, String> {
    restamp(file, LAKE_LOG_FORMAT_VERSION_1_0, LAKE_LOG_FORMAT_VERSION)
}

/// Re-stamps a commit record at 1.0 when a 1.0 reader can read every entry
/// it holds, and refuses one that carries a schema id on an added file or a
/// column type history, which 1.0 has no tag for
pub fn log_1_1_to_1_0(file: &[u8]) -> Result<Vec<u8>, String> {
    let data = VersionFileData::decode(file, "log migration").map_err(|e| e.to_string())?;
    for entry in &data.entries {
        let carried = match entry {
            LogEntry::AddFile(added) => added.schema_id != 0,
            LogEntry::AddIndexFile(added) => added.file.schema_id != 0,
            LogEntry::TypeHistory(_) => true,
            _ => false,
        };
        if carried {
            return Err(format!(
                "version {} carries a file's schema id or a column type history, which a 1.0 \
                 record has no tag for",
                data.header.version
            ));
        }
    }
    restamp(file, LAKE_LOG_FORMAT_VERSION, LAKE_LOG_FORMAT_VERSION_1_0)
}

/// Names `to` in the envelope version of a commit record at `from` and
/// recomputes the checksum that covers its header. The entry section and
/// its own checksum are untouched
fn restamp(file: &[u8], from: FormatVersion, to: FormatVersion) -> Result<Vec<u8>, String> {
    let (kind, version) = envelope::peek(file).map_err(|e| e.to_string())?;
    if kind != FormatKind::LakeTransactionLog {
        return Err(format!("a {kind} file, not a lake commit record"));
    }
    if version != from {
        return Err(format!(
            "at version {version}, this step moves {from} forward"
        ));
    }
    if file.len() < COMMIT_HEADER_LEN {
        return Err(format!(
            "commit header needs {COMMIT_HEADER_LEN} bytes, got {}",
            file.len()
        ));
    }
    let mut out = file.to_vec();
    out[4..8].copy_from_slice(&to.to_le_bytes());
    out[8..12].fill(0);
    let crc = crc32fast::hash(&out[..COMMIT_HEADER_LEN]);
    out[8..12].copy_from_slice(&crc.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::LAKE_LOG_FIXTURE_1_0;

    #[test]
    fn test_a_1_0_record_moves_forward_and_back_with_its_entries_intact() {
        let before = VersionFileData::decode(LAKE_LOG_FIXTURE_1_0, "fixture").expect("1.0");
        let moved = log_1_0_to_1_1(LAKE_LOG_FIXTURE_1_0).expect("forward");
        let (_, version) = envelope::peek(&moved).expect("peek");
        assert_eq!(version, LAKE_LOG_FORMAT_VERSION);
        let after = VersionFileData::decode(&moved, "moved").expect("1.1 decodes");
        assert_eq!(after.entries, before.entries);
        assert_eq!(after.header, before.header);
        let back = log_1_1_to_1_0(&moved).expect("backward");
        assert_eq!(back, LAKE_LOG_FIXTURE_1_0);
    }

    #[test]
    fn test_a_record_at_the_wrong_version_is_refused_by_each_step() {
        let refused = log_1_1_to_1_0(LAKE_LOG_FIXTURE_1_0).expect_err("a 1.0 record is not 1.1");
        assert!(refused.contains("at version 1.0"), "{refused}");
        let moved = log_1_0_to_1_1(LAKE_LOG_FIXTURE_1_0).expect("forward");
        let refused = log_1_0_to_1_1(&moved).expect_err("a 1.1 record is not 1.0");
        assert!(refused.contains("at version 1.1"), "{refused}");
    }
}

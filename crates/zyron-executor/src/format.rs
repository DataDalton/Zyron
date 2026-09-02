//! The replication apply log format registration.
//!
//! A changeset chunk is the unit the apply log records and a follower
//! replays. Chunks travel through consensus and are replayed from the log
//! the consensus group holds, so during a rolling upgrade one log holds
//! chunks written by nodes at two versions. Each chunk carries its own
//! version tag in its first byte for that reason, and both versions coexist
//! rather than being rewritten in place

use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::format::{FormatKind, RecordVersion};

/// Version tag every changeset chunk carries in its first byte
pub const APPLY_RECORD_VERSION: RecordVersion = RecordVersion::V1;

/// The same tag as the raw byte the chunk header holds
pub const APPLY_RECORD_VERSION_BYTE: u8 = APPLY_RECORD_VERSION.get();

/// The format version the registry declares, which tracks the record tag
pub const APPLY_LOG_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::ReplicationApplyLog,
        writer_current_version: APPLY_LOG_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(APPLY_LOG_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: "0.11.0",
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "changeset chunks with a per-chunk version tag, versions coexist in the log",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_apply_log_registers_once() {
        let count = inventory::iter::<FormatRegistration>
            .into_iter()
            .filter(|r| r.kind == FormatKind::ReplicationApplyLog)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_the_record_tag_matches_the_registered_version() {
        assert_eq!(
            APPLY_RECORD_VERSION.as_format_version().minor,
            APPLY_LOG_FORMAT_VERSION.major as u16,
            "the chunk tag and the registered version move together"
        );
    }
}

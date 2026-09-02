//! Raft format registrations.
//!
//! The log file carries the envelope in its header and every entry carries a
//! version tag, so one log holds entries written across a rolling upgrade
//! and a follower's copy of an entry is byte identical to the leader's. The
//! log is a historical record consensus still replays, so entry versions
//! coexist and new entries are written at the current version.
//!
//! The snapshot pointer is rewritten whole every time a snapshot is taken,
//! so it migrates eagerly with the next one

use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::format::{FormatKind, RecordVersion};

const GATE: &str = "0.11.0";

/// Version the log file header is written at
pub const RAFT_LOG_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version tag every log entry carries
pub const RAFT_ENTRY_RECORD_VERSION: RecordVersion = RecordVersion::V1;

/// Version the snapshot pointer and its chunk stream are written at
pub const SNAPSHOT_TRANSFER_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::RaftLog,
        writer_current_version: RAFT_LOG_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(RAFT_LOG_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "envelope header plus a per-entry version tag, entries coexist in the log",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::SnapshotTransfer,
        writer_current_version: SNAPSHOT_TRANSFER_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(SNAPSHOT_TRANSFER_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "snapshot pointer and chunk stream, eager on the next snapshot",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raft_formats_register_exactly_once() {
        for kind in [FormatKind::RaftLog, FormatKind::SnapshotTransfer] {
            let count = inventory::iter::<FormatRegistration>
                .into_iter()
                .filter(|r| r.kind == kind)
                .count();
            assert_eq!(count, 1, "{kind} submitted {count} registrations");
        }
    }
}

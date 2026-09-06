//! Raft format registrations.
//!
//! The log file carries the envelope in its header and every entry carries a
//! version tag, so one log holds entries written across a rolling upgrade
//! and a follower's copy of an entry is byte identical to the leader's. The
//! log is a historical record consensus still replays, so entry versions
//! coexist and new entries are written at the current version.
//!
//! The snapshot pointer is rewritten whole every time a snapshot is taken,
//! so it migrates eagerly with the next one.
//!
//! The consensus protocol, the frames between members of a group, registers
//! here as well, so the release check and the protocol versions view report
//! it beside the client and mesh protocols

use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::format::wire_version::{WireProtocol, WireProtocolVersion, WireVersionStatus};
use zyron_common::format::{FormatKind, RecordVersion};

use crate::transport::CONSENSUS_PROTOCOL_VERSION;

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

inventory::submit! {
    WireProtocolVersion {
        protocol: WireProtocol::Consensus,
        version: CONSENSUS_PROTOCOL_VERSION as u32,
        status: WireVersionStatus::Current,
        introduced_in_binary_version: "0.8.0",
        retired_in_binary_version: None,
        notes: "twenty byte frame header naming the version, five message kinds, a field is \
                added by appending it and an absent trailing field reads as its default",
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

    #[test]
    fn test_the_consensus_protocol_registers_its_frame_version_once() {
        let rows: Vec<_> = inventory::iter::<WireProtocolVersion>
            .into_iter()
            .filter(|v| v.protocol == WireProtocol::Consensus)
            .collect();
        assert_eq!(rows.len(), 1, "the consensus protocol registers once");
        assert_eq!(rows[0].version, u32::from(CONSENSUS_PROTOCOL_VERSION));
        assert_eq!(rows[0].status, WireVersionStatus::Current);
    }
}

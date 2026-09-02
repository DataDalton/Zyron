//! CDC format registrations.
//!
//! A snapshot manifest is a point-in-time capture that a consumer replays
//! from, so old and new versions coexist and a manifest is only moved
//! forward when it is read. The stream checkpoint is rewritten on every
//! progress update, so it migrates eagerly the next time a stream restarts

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};

const GATE: &str = "0.11.0";

/// Version the snapshot manifest is written at
pub const CDC_SNAPSHOT_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version the stream checkpoint is written at
pub const CDC_CHECKPOINT_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::Snapshot,
        writer_current_version: CDC_SNAPSHOT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CDC_SNAPSHOT_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "point-in-time capture of a table set, historical, migrated on read",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::StreamingCdcCheckpoint,
        writer_current_version: CDC_CHECKPOINT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CDC_CHECKPOINT_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "per-stream progress offsets, eager on stream restart",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdc_formats_register_exactly_once() {
        for kind in [FormatKind::Snapshot, FormatKind::StreamingCdcCheckpoint] {
            let count = inventory::iter::<FormatRegistration>
                .into_iter()
                .filter(|r| r.kind == kind)
                .count();
            assert_eq!(count, 1, "{kind} submitted {count} registrations");
        }
    }
}

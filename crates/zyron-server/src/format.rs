//! Server format registrations.
//!
//! Two of these are text files a person is meant to read, so they carry the
//! envelope as a declared `[format]` section rather than as a binary header.
//! The section says the same thing the header says, and a version outside
//! the reader window fails closed the same way.
//!
//! A backup is a point-in-time copy that older clusters may still need to
//! restore, so its versions coexist and downgrade-write is available for
//! exporting to one. The config file is rewritten on every `ALTER SYSTEM`,
//! so it migrates eagerly on startup, before anything reads a value out of
//! it

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};

const GATE: &str = "0.11.0";

/// Version a backup manifest is written at
pub const BACKUP_ARCHIVE_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version `zyron.toml` is written at
pub const CONFIG_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::BackupArchive,
        writer_current_version: BACKUP_ARCHIVE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(BACKUP_ARCHIVE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: true,
        notes: "file set and checksums, historical, migrated on restore",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::ZyronTomlConfig,
        writer_current_version: CONFIG_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CONFIG_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "server configuration, eager on startup before any value is read",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_formats_register_exactly_once() {
        for kind in [FormatKind::BackupArchive, FormatKind::ZyronTomlConfig] {
            let count = inventory::iter::<FormatRegistration>
                .into_iter()
                .filter(|r| r.kind == kind)
                .count();
            assert_eq!(count, 1, "{kind} submitted {count} registrations");
        }
    }

    #[test]
    fn test_a_backup_can_be_written_for_an_older_cluster() {
        let registration = inventory::iter::<FormatRegistration>
            .into_iter()
            .find(|r| r.kind == FormatKind::BackupArchive)
            .expect("registered");
        assert!(registration.downgrade_write_supported);
        assert_eq!(registration.migration_policy, MigrationPolicy::Coexist);
    }
}

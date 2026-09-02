//! Lake format registrations.
//!
//! A manifest is rewritten in full on every commit, so it migrates eagerly:
//! the next commit after an upgrade writes the current version and the old
//! one leaves the table with the snapshot that named it. Index artifacts are
//! rebuilt rather than edited, so they migrate lazily on the next rebuild.
//! Delete predicates and transaction log entries are historical records that
//! time travel still reads, so both versions coexist

use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::format::{FormatKind, RecordVersion};

/// The Zyron version these formats were last bumped in
const GATE: &str = "0.11.0";

/// Version manifests are written at
pub const LAKE_MANIFEST_FORMAT_VERSION: FormatVersion = FormatVersion::new(2, 0);

/// Version index artifacts are written at
pub const LAKE_INDEX_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version delete predicate records are written at
pub const DELETE_PREDICATE_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version transaction log commit records are written at
pub const LAKE_LOG_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Per-record version tag every commit record in the log carries
pub const LAKE_LOG_RECORD_VERSION: RecordVersion = RecordVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::LakeManifest,
        writer_current_version: LAKE_MANIFEST_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(LAKE_MANIFEST_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "file set, stats and delete predicates, eager on the next commit",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::LakeIndex,
        writer_current_version: LAKE_INDEX_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(LAKE_INDEX_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "per-file value to row mappings, lazy on the next index rebuild",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::DeletePredicate,
        writer_current_version: DELETE_PREDICATE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(DELETE_PREDICATE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "predicate deletes readers filter live files through, historical, coexist",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::LakeTransactionLog,
        writer_current_version: LAKE_LOG_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(LAKE_LOG_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "commit records with a per-record version tag, historical, coexist",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAKE_KINDS: &[FormatKind] = &[
        FormatKind::LakeManifest,
        FormatKind::LakeIndex,
        FormatKind::DeletePredicate,
        FormatKind::LakeTransactionLog,
    ];

    #[test]
    fn test_every_lake_format_registers_exactly_once() {
        for kind in LAKE_KINDS {
            let count = inventory::iter::<FormatRegistration>
                .into_iter()
                .filter(|r| r.kind == *kind)
                .count();
            assert_eq!(count, 1, "{kind} submitted {count} registrations");
        }
    }

    #[test]
    fn test_historical_lake_formats_coexist() {
        for kind in [FormatKind::DeletePredicate, FormatKind::LakeTransactionLog] {
            let registration = inventory::iter::<FormatRegistration>
                .into_iter()
                .find(|r| r.kind == kind)
                .expect("registered");
            assert_eq!(registration.migration_policy, MigrationPolicy::Coexist);
        }
    }
}

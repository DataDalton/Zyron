//! Registrations for formats whose owning subsystem ships in a later phase.
//!
//! A magic is allocated for the life of the product the moment it is
//! reserved, so nothing else can take it and any tool that meets one of
//! these files identifies it correctly. The envelope helpers serve these
//! kinds today, which is what `zyron-ctl format inspect` reads them with.
//!
//! When the owning subsystem lands, its registration moves into that crate
//! beside the writer and the entry here is deleted. Two registrations for
//! one kind refuse the load, so the move cannot be half done

use super::kind::FormatKind;
use super::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use super::version::{FormatVersion, VersionWindow};

/// The Zyron version these reservations were made in
const RESERVED_IN: &str = "0.11.0";

/// Builds a first-version registration for a reserved kind
const fn reserved(
    kind: FormatKind,
    policy: MigrationPolicy,
    notes: &'static str,
) -> FormatRegistration {
    FormatRegistration {
        kind,
        writer_current_version: FormatVersion::V1,
        reader_supported_versions: VersionWindow::single(FormatVersion::V1),
        migration_policy: policy,
        migration_reversible: true,
        binary_version_gate: RESERVED_IN,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes,
    }
}

inventory::submit! {
    reserved(
        FormatKind::VolumeMetadata,
        MigrationPolicy::Lazy,
        "magic allocated, written by the volume store when it lands",
    )
}

inventory::submit! {
    reserved(
        FormatKind::PromptRegistryStorage,
        MigrationPolicy::Lazy,
        "magic allocated, written by the prompt registry when it lands",
    )
}

inventory::submit! {
    reserved(
        FormatKind::WorkflowDefinitionOnDisk,
        MigrationPolicy::Lazy,
        "magic allocated, written by workflow persistence when it lands",
    )
}

inventory::submit! {
    reserved(
        FormatKind::AppImageBundle,
        MigrationPolicy::Coexist,
        "magic allocated, written by the App bundle builder when it lands",
    )
}

/// The kinds reserved here, which the startup report prints separately from
/// the kinds that have a live writer
pub const RESERVED_KINDS: &[FormatKind] = &[
    FormatKind::VolumeMetadata,
    FormatKind::PromptRegistryStorage,
    FormatKind::WorkflowDefinitionOnDisk,
    FormatKind::AppImageBundle,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_reserved_kind_submits_exactly_one_registration() {
        for kind in RESERVED_KINDS {
            let count = inventory::iter::<FormatRegistration>
                .into_iter()
                .filter(|r| r.kind == *kind)
                .count();
            assert_eq!(count, 1, "{kind} has {count} registrations");
        }
    }

    #[test]
    fn test_reserved_kinds_start_at_version_one() {
        for registration in inventory::iter::<FormatRegistration> {
            if RESERVED_KINDS.contains(&registration.kind) {
                assert_eq!(registration.writer_current_version, FormatVersion::V1);
                assert_eq!(
                    registration.reader_supported_versions,
                    VersionWindow::single(FormatVersion::V1)
                );
            }
        }
    }
}

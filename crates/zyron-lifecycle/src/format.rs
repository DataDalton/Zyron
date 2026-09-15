//! Format registrations for the compliance log and the verification
//! artifacts.
//!
//! The compliance log's rows live in a catalog table rather than in a file
//! of its own, so what is versioned there is the entry encoding. Each entry
//! carries its version tag and an entry is never rewritten, so entries of
//! different versions coexist in one table.
//!
//! A commit chain is a run of fixed records behind an envelope header. It is
//! appended to and never rewritten, and each record's integrity is the chain
//! itself: a record states the link of the one before it. The anchor store
//! and the verification history are small and rewritten whole, so both
//! migrate eagerly on the next write.

use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::format::{FormatKind, RecordVersion};

use crate::verify::CHAIN_FORMAT_VERSION;
use crate::verify::RUN_LOG_FORMAT_VERSION;
use crate::verify::anchor::ANCHOR_FORMAT_VERSION;

/// Version tag every compliance log entry carries
pub const AUDIT_RECORD_VERSION: RecordVersion = RecordVersion::V1;

/// The same tag as the raw byte an entry stores
pub const AUDIT_RECORD_VERSION_BYTE: u8 = AUDIT_RECORD_VERSION.get();

/// The format version the registry declares for the entry encoding
pub const AUDIT_RECORD_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// The release the verification formats were introduced in
const VERIFY_GATE: &str = "0.19.0";

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::ComplianceLogRecord,
        writer_current_version: AUDIT_RECORD_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(AUDIT_RECORD_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: "0.11.0",
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "compliance events with a per-entry version tag, never rewritten",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::VerifiableCommitChain,
        writer_current_version: CHAIN_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(CHAIN_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: VERIFY_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "fixed commit entries appended in order, each carrying the link of the one before it",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::VerificationAnchor,
        writer_current_version: ANCHOR_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(ANCHOR_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: VERIFY_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "anchored chain heads, rewritten in full at the next anchor",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::VerificationRunLog,
        writer_current_version: RUN_LOG_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(RUN_LOG_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: VERIFY_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "verification history, rewritten in full after the next run",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registrations(kind: FormatKind) -> usize {
        inventory::iter::<FormatRegistration>
            .into_iter()
            .filter(|r| r.kind == kind)
            .count()
    }

    #[test]
    fn test_each_format_registers_once() {
        for kind in [
            FormatKind::ComplianceLogRecord,
            FormatKind::VerifiableCommitChain,
            FormatKind::VerificationAnchor,
            FormatKind::VerificationRunLog,
        ] {
            assert_eq!(registrations(kind), 1, "{}", kind.catalog_name());
        }
    }
}

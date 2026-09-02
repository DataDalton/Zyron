//! The audit hash chain format registration.
//!
//! The durable chain lives in the catalog's compliance log table rather than
//! in a file of its own, so what is versioned here is the entry encoding.
//! Each entry carries its version tag and the tag is folded into the chained
//! hash, so an entry written before an upgrade verifies unchanged after one.
//!
//! The chain is an immutable historical record. Rewriting an entry would
//! break every hash after it, so entries of different versions coexist and
//! nothing is ever migrated in place

use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::format::{FormatKind, RecordVersion};

/// Version tag every audit entry carries
pub const AUDIT_RECORD_VERSION: RecordVersion = RecordVersion::V1;

/// The same tag as the raw byte an entry stores
pub const AUDIT_RECORD_VERSION_BYTE: u8 = AUDIT_RECORD_VERSION.get();

/// The format version the registry declares
pub const AUDIT_CHAIN_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::AuditHashChain,
        writer_current_version: AUDIT_CHAIN_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(AUDIT_CHAIN_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Coexist,
        migration_reversible: true,
        binary_version_gate: "0.11.0",
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "tamper-evident entries with a per-entry version tag, never rewritten",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit_chain::AuditChain;

    #[test]
    fn test_the_audit_chain_registers_once() {
        let count = inventory::iter::<FormatRegistration>
            .into_iter()
            .filter(|r| r.kind == FormatKind::AuditHashChain)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_entries_carry_the_version_tag_and_still_chain() {
        let chain = AuditChain::new();
        let first = chain.next_entry(0, "subject".into(), 1, 100, "detail".into());
        let second = chain.next_entry(1, "subject".into(), 1, 200, "detail".into());
        assert_eq!(first.record_version, AUDIT_RECORD_VERSION_BYTE);
        assert_eq!(second.record_version, AUDIT_RECORD_VERSION_BYTE);
        assert_eq!(second.prev_hash, first.entry_hash);
        let (verified, intact) = AuditChain::verify(&[first, second]);
        assert_eq!(verified, 2);
        assert!(intact);
    }

    #[test]
    fn test_the_version_tag_is_covered_by_the_chained_hash() {
        let chain = AuditChain::new();
        let mut entry = chain.next_entry(0, "subject".into(), 1, 100, "detail".into());
        let original = entry.entry_hash;
        entry.record_version = entry.record_version.wrapping_add(1);
        assert_ne!(
            entry.compute_hash(),
            original,
            "changing the tag has to change the hash"
        );
    }
}

//! The out-of-line extended value format registration.
//!
//! A stored object holds one value that was too large to keep inline. Objects
//! are content addressed, so a migration produces a different address and the
//! old object is unreachable once nothing references it. Migration is lazy
//! for that reason: an object moves forward the next time its value is
//! rewritten, and the reference count sweep reclaims what is left behind

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};

/// Version stored objects are written at
pub const TOAST_OBJECT_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::Toast,
        writer_current_version: TOAST_OBJECT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(TOAST_OBJECT_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: "0.11.0",
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "content addressed object with a format stamp, lazy on next rewrite",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_object_format_registers_once() {
        let count = inventory::iter::<FormatRegistration>
            .into_iter()
            .filter(|r| r.kind == FormatKind::Toast)
            .count();
        assert_eq!(count, 1);
    }
}

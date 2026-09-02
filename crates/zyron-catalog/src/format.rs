//! Catalog format registrations.
//!
//! ANALYZE rewrites a table's statistics file in full, so the format
//! migrates eagerly: the next ANALYZE after an upgrade writes the current
//! version and nothing has to sweep. A file behind the current version is
//! still read until then, so plans keep their estimates across the upgrade

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::VersionWindow;

use crate::statistics::STATISTICS_FORMAT_VERSION;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::StatisticsFile,
        writer_current_version: STATISTICS_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(STATISTICS_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: true,
        binary_version_gate: "0.11.0",
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "per-column histograms and cardinalities, eager on the next ANALYZE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_statistics_format_registers_once() {
        let count = inventory::iter::<FormatRegistration>
            .into_iter()
            .filter(|r| r.kind == FormatKind::StatisticsFile)
            .count();
        assert_eq!(count, 1);
    }
}

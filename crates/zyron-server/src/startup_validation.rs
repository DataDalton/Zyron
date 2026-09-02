//! Startup validation of the format substrate.
//!
//! A binary that cannot say which format versions it reads has no safe way
//! to open a data directory, so this runs before anything is opened and
//! refuses to start rather than discovering the gap at the first read.
//!
//! Everything checked here is also checked by the release check, so a
//! release that ships cannot fail this. The gate exists because the release
//! check runs against the tree and this runs against the binary that is
//! actually about to open your data

use zyron_common::format::{FormatSubstrate, MAGIC_ALLOCATIONS};
use zyron_common::{Result, ZyronError};

/// What the gate found, for the line the server logs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub formats: usize,
    pub reserved_formats: usize,
    pub schemes: usize,
    pub catalog_tables: usize,
    pub deprecations: usize,
    pub wire_versions: usize,
    pub rewriters: usize,
    pub current_wire_version: u32,
    pub channel: String,
    pub auto_upgrade_enabled: bool,
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "format substrate ready, {} formats ({} reserved), {} signature schemes, \
             {} catalog tables, {} deprecations, {} user-object rewriters, \
             wire version {} of {}, upgrade channel {}, auto_upgrade_enabled {}",
            self.formats,
            self.reserved_formats,
            self.schemes,
            self.catalog_tables,
            self.deprecations,
            self.rewriters,
            self.current_wire_version,
            self.wire_versions,
            self.channel,
            self.auto_upgrade_enabled
        )
    }
}

/// Loads the substrate and refuses to go on if anything is missing.
///
/// The checks, in the order a failure is most likely to matter:
/// every format kind has a registration, no two formats share a magic, every
/// version bump inside a reader window has a migrator and a fixture (which
/// the registry load itself enforces), every catalog table's evolution chain
/// is reachable from its first version, and every deprecation has a
/// lifecycle in order
pub fn validate() -> Result<ValidationReport> {
    let substrate = zyron_common::format::substrate()?;
    check_magic_uniqueness()?;
    substrate.verify_complete()?;
    check_rewriters(substrate)?;
    Ok(report(substrate))
}

/// Builds the report without re-running the checks, for a caller that has
/// already validated
pub fn report(substrate: &FormatSubstrate) -> ValidationReport {
    let settings = zyron_common::format::upgrade_board().settings();
    ValidationReport {
        formats: substrate.formats.len(),
        reserved_formats: zyron_common::format::reserved::RESERVED_KINDS.len(),
        schemes: substrate.schemes.schemes().len(),
        catalog_tables: substrate.catalog_schemas.tables().len(),
        deprecations: substrate.deprecations.records().len(),
        wire_versions: substrate.wire_versions.versions().len(),
        rewriters: zyron_parser::rewriter::registered().len(),
        current_wire_version: substrate.wire_versions.current_version(),
        channel: settings.channel.label().to_string(),
        auto_upgrade_enabled: settings.auto_upgrade_enabled,
    }
}

/// Two formats sharing a magic means a file cannot be identified from its
/// first four bytes, which every reader depends on
fn check_magic_uniqueness() -> Result<()> {
    for (index, left) in MAGIC_ALLOCATIONS.iter().enumerate() {
        for right in MAGIC_ALLOCATIONS.iter().skip(index + 1) {
            if left.magic == right.magic {
                return Err(ZyronError::FormatRegistry(format!(
                    "formats `{}` and `{}` both claim magic {}. A magic is allocated once \
                     for the life of the product",
                    left.kind,
                    right.kind,
                    zyron_common::format::envelope::printable_magic(&left.magic)
                )));
            }
        }
    }
    Ok(())
}

/// Every user-authored SQL deprecation needs a registered rewriter, so an
/// upgrade can say what it would do to an object rather than only that the
/// object will break
fn check_rewriters(substrate: &FormatSubstrate) -> Result<()> {
    let rewriters = zyron_parser::rewriter::registered();
    for record in substrate.deprecations.records() {
        if !record.item_kind.is_user_authored_sql() {
            continue;
        }
        let covered = rewriters.iter().any(|rewrite| {
            rewrite.description.contains(record.item_id)
                || rewrite.name.contains(record.item_id)
                || record
                    .replacement_ref
                    .map(|replacement| rewrite.description.contains(replacement))
                    .unwrap_or(false)
        });
        if !covered {
            return Err(ZyronError::FormatRegistry(format!(
                "the deprecation of `{}` is a SQL surface with no registered rewriter. An \
                 upgrade cannot tell an operator what it would do to the objects that use it",
                record.item_id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_running_binary_passes_the_gate() {
        let report = validate().expect("this binary is complete");
        assert_eq!(
            report.formats,
            zyron_common::format::ALL_FORMAT_KINDS.len(),
            "the server links every crate, so every format kind is registered"
        );
        assert!(
            report.schemes >= 3,
            "the three classical schemes are registered"
        );
        assert!(report.catalog_tables > 0);
        assert_eq!(report.current_wire_version, 3);
    }

    #[test]
    fn test_the_report_reads_as_one_line() {
        let substrate = zyron_common::format::substrate().expect("loads");
        let text = report(substrate).to_string();
        assert!(text.contains("format substrate ready"), "{text}");
        assert!(text.contains("signature schemes"), "{text}");
        assert!(text.contains("upgrade channel"), "{text}");
    }

    #[test]
    fn test_magics_are_unique_in_this_binary() {
        check_magic_uniqueness().expect("no duplicate magics");
    }

    #[test]
    fn test_every_registered_format_is_reachable_from_its_magic() {
        let substrate = zyron_common::format::substrate().expect("loads");
        for entry in substrate.formats.entries() {
            let kind = entry.registration.kind;
            assert_eq!(
                zyron_common::format::FormatKind::from_magic(kind.magic()),
                Some(kind)
            );
        }
    }
}

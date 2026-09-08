//! Migrations that read the catalog rather than a file body.
//!
//! Most format changes are answered by a migrator that takes one body's bytes
//! and returns the next version's bytes. Some are not: a version bump that
//! gives an existing field a meaning needs no byte to move, but it does need
//! state recorded once, in the catalog, before any reader relies on the new
//! meaning.
//!
//! The registration is data and lives here beside the other format registries.
//! The runner lives with the layer that can open a catalog, the same split
//! catalog schema evolution uses, because only that layer knows how to read
//! and write a table entry.

use super::kind::FormatKind;
use super::version::FormatVersion;

/// One catalog-scan migration, run once when a data directory moves to the
/// version it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogScanMigration {
    /// The format whose version bump this scan belongs to
    pub kind: FormatKind,
    /// The version the directory is moving from
    pub from: FormatVersion,
    /// The version the scan completes the move to
    pub to: FormatVersion,
    /// The Zyron version that introduced the scan
    pub introduced_in_binary_version: &'static str,
    /// True when the change cannot be undone, which is what makes a
    /// downgrade past it refuse by name rather than silently produce a
    /// directory an older binary misreads
    pub one_way: bool,
    /// The name the scan reports itself by, printed by the downgrade refusal
    /// and by the progress log
    pub scan_name: &'static str,
    /// One line naming what the scan records
    pub description: &'static str,
}

inventory::collect!(CatalogScanMigration);

/// Every registered catalog scan, ordered by the version it completes.
pub fn catalog_scan_migrations() -> Vec<CatalogScanMigration> {
    let mut out: Vec<CatalogScanMigration> = inventory::iter::<CatalogScanMigration>
        .into_iter()
        .copied()
        .collect();
    out.sort_by(|a, b| {
        a.kind
            .catalog_name()
            .cmp(b.kind.catalog_name())
            .then(a.to.cmp(&b.to))
    });
    out
}

/// The scan that completes a move to `to` for one format, or None when the
/// version bump needs no catalog work.
pub fn catalog_scan_for(kind: FormatKind, to: FormatVersion) -> Option<CatalogScanMigration> {
    inventory::iter::<CatalogScanMigration>
        .into_iter()
        .find(|m| m.kind == kind && m.to == to)
        .copied()
}

/// Why a downgrade past a one-way scan is refused, naming the scan so an
/// operator can find it rather than being told only that it failed.
pub fn downgrade_refusal(kind: FormatKind, to: FormatVersion) -> Option<String> {
    let scan = catalog_scan_for(kind, to)?;
    if !scan.one_way {
        return None;
    }
    Some(format!(
        "{} cannot move back below {} because the {} scan is one way: {}. Restore from a backup \
         taken before the upgrade instead",
        kind.catalog_name(),
        scan.to,
        scan.scan_name,
        scan.description
    ))
}

/// One paragraph of on-disk documentation for a format at one version,
/// surfaced by `zyron_sys.storage.format_documentation`.
///
/// The layout columns of that view are generated from the framing, which says
/// where the bytes are but not what a reader is supposed to make of them. This
/// carries the sentence a version bump exists to communicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatDocumentation {
    pub kind: FormatKind,
    pub version: FormatVersion,
    pub text: &'static str,
}

inventory::collect!(FormatDocumentation);

/// The documentation recorded for one format version, empty when none is.
pub fn documentation_for(kind: FormatKind, version: FormatVersion) -> Option<&'static str> {
    inventory::iter::<FormatDocumentation>
        .into_iter()
        .find(|d| d.kind == kind && d.version == version)
        .map(|d| d.text)
}

//! Data-residency enforcement. Honors the `data_residency` table option
//! (defined in the enterprise-distribution plan) when selecting/validating a
//! tier backend region. A tier migration that would move data out of the
//! required region is rejected.

use zyron_catalog::schema::TableEntry;
use zyron_common::{Result, ZyronError};

/// Extracts the region implied by a tier destination URI. For object stores
/// the region is taken from an explicit `region=` query hint or, failing
/// that, an `s3.<region>.` / `<region>.` host segment. Returns None when the
/// destination carries no region (e.g. local fs).
pub fn region_of_destination(uri: &str) -> Option<String> {
    if let Some(idx) = uri.find("region=") {
        let rest = &uri[idx + 7..];
        let end = rest.find(['&', '/', ' ']).unwrap_or(rest.len());
        let r = &rest[..end];
        if !r.is_empty() {
            return Some(r.to_string());
        }
    }
    None
}

/// Asserts a tier migration to `destination_uri` keeps the table within its
/// required residency region. No required region => always allowed.
pub fn assert_residency(entry: &TableEntry, destination_uri: &str) -> Result<()> {
    let required = entry.lifecycle.residency_region.trim();
    if required.is_empty() {
        return Ok(());
    }
    match region_of_destination(destination_uri) {
        Some(actual) if actual.eq_ignore_ascii_case(required) => Ok(()),
        Some(actual) => Err(ZyronError::RetentionViolation(format!(
            "data residency '{required}' required but destination is in '{actual}'"
        ))),
        None => Err(ZyronError::RetentionViolation(format!(
            "data residency '{required}' required but destination '{destination_uri}' has no region"
        ))),
    }
}

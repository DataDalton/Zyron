//! Compliance profile presets. `WITH (compliance_profile='gdpr'|'hipaa'|'sox')`
//! expands to a vetted set of lifecycle table options applied before other
//! option handling.

use zyron_common::{Result, ZyronError};

/// Expands a compliance profile name into concrete (key, value) table options.
/// The dispatch layer merges these in before user-specified options so the
/// user can still override a specific key.
pub fn expand_compliance_profile(profile: &str) -> Result<Vec<(String, String)>> {
    let kv = |k: &str, v: &str| (k.to_string(), v.to_string());
    match profile.to_ascii_lowercase().as_str() {
        "gdpr" => Ok(vec![
            kv("soft_delete", "true"),
            kv("purge_after_soft_delete", "30 days"),
            kv("archive_on_purge", "true"),
            kv("audit", "true"),
        ]),
        "hipaa" => Ok(vec![
            kv("soft_delete", "true"),
            kv("retention_lock", "6 years"),
            kv("audit", "true"),
            kv("classification", "restricted"),
        ]),
        "sox" => Ok(vec![
            kv("retention_lock", "7 years"),
            kv("immutable", "true"),
            kv("audit", "true"),
        ]),
        other => Err(ZyronError::Internal(format!(
            "unknown compliance_profile: {other}"
        ))),
    }
}

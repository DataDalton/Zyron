//! Semantic versioning type and operations.
//!
//! SemVer is stored as a packed u64: major(21 bits) | minor(21 bits) | patch(21 bits) | pre(1 bit).
//! This gives proper numeric ordering: 1.2.3 < 1.10.0 (not lexicographic).
//! Max version component is 2^21 - 1 = 2,097,151.

use zyron_common::{Result, ZyronError};

const MAJOR_SHIFT: u32 = 43;
const MINOR_SHIFT: u32 = 22;
const PATCH_SHIFT: u32 = 1;
const COMPONENT_MASK: u64 = 0x1FFFFF; // 21 bits
const PRE_FLAG: u64 = 1;
const MAX_COMPONENT: u64 = 0x1FFFFF; // 2,097,151

/// Parses a SemVer string like "1.2.3" or "1.2.3-rc.1" into packed u64.
/// Pre-release versions sort before their release counterpart.
pub fn semver_parse(text: &str) -> Result<u64> {
    let trimmed = text.trim();
    let trimmed = trimmed.strip_prefix('v').unwrap_or(trimmed);

    let (version_part, has_pre) = if let Some(idx) = trimmed.find('-') {
        (&trimmed[..idx], true)
    } else {
        (trimmed, false)
    };

    let parts: Vec<&str> = version_part.split('.').collect();
    if parts.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "Invalid SemVer format '{}', expected 'major.minor.patch'",
            text
        )));
    }

    let major = parts[0].parse::<u64>().map_err(|e| {
        ZyronError::ExecutionError(format!("Invalid major version '{}': {}", parts[0], e))
    })?;
    let minor = parts[1].parse::<u64>().map_err(|e| {
        ZyronError::ExecutionError(format!("Invalid minor version '{}': {}", parts[1], e))
    })?;
    let patch = parts[2].parse::<u64>().map_err(|e| {
        ZyronError::ExecutionError(format!("Invalid patch version '{}': {}", parts[2], e))
    })?;

    if major > MAX_COMPONENT || minor > MAX_COMPONENT || patch > MAX_COMPONENT {
        return Err(ZyronError::ExecutionError(format!(
            "Version component exceeds maximum ({})",
            MAX_COMPONENT
        )));
    }

    let mut packed = (major << MAJOR_SHIFT) | (minor << MINOR_SHIFT) | (patch << PATCH_SHIFT);
    if has_pre {
        // Pre-release flag: set to 0 so pre-release sorts BEFORE release.
        // Release versions have the pre-bit set to 1.
        // This way: 1.0.0-rc < 1.0.0
    } else {
        packed |= PRE_FLAG;
    }

    Ok(packed)
}

/// Formats a packed u64 back to a SemVer string.
/// Pre-release versions include a "-pre" suffix (the specific pre-release
/// tag is not stored in the packed representation).
pub fn semver_format(packed: u64) -> String {
    let major = (packed >> MAJOR_SHIFT) & COMPONENT_MASK;
    let minor = (packed >> MINOR_SHIFT) & COMPONENT_MASK;
    let patch = (packed >> PATCH_SHIFT) & COMPONENT_MASK;
    let is_release = (packed & PRE_FLAG) != 0;

    if is_release {
        format!("{}.{}.{}", major, minor, patch)
    } else {
        format!("{}.{}.{}-pre", major, minor, patch)
    }
}

/// Extracts the major version component.
pub fn semver_major(packed: u64) -> u32 {
    ((packed >> MAJOR_SHIFT) & COMPONENT_MASK) as u32
}

/// Extracts the minor version component.
pub fn semver_minor(packed: u64) -> u32 {
    ((packed >> MINOR_SHIFT) & COMPONENT_MASK) as u32
}

/// Extracts the patch version component.
pub fn semver_patch(packed: u64) -> u32 {
    ((packed >> PATCH_SHIFT) & COMPONENT_MASK) as u32
}

/// Returns true if this is a pre-release version.
pub fn semver_is_prerelease(packed: u64) -> bool {
    (packed & PRE_FLAG) == 0
}

/// Compares two packed SemVer values.
/// Returns -1 if a < b, 0 if equal, 1 if a > b.
/// Natural numeric ordering: 1.2.3 < 1.10.0.
pub fn semver_compare(a: u64, b: u64) -> i32 {
    if a < b {
        -1
    } else if a > b {
        1
    } else {
        0
    }
}

/// Checks if a version satisfies a constraint string.
/// Supported constraints:
/// - "^1.2.0": compatible with 1.2.0 (same major, >= minor.patch)
/// - "~1.2.0": approximately 1.2.0 (same major.minor, >= patch)
/// - ">=1.0.0": greater than or equal to
/// - "<=2.0.0": less than or equal to
/// - ">1.0.0": strictly greater than
/// - "<2.0.0": strictly less than
/// - "=1.2.3" or "1.2.3": exact match
/// - ">=1.0.0 <2.0.0": range (space-separated, all must match)
pub fn semver_satisfies(version: u64, constraint: &str) -> Result<bool> {
    let trimmed = constraint.trim();

    // Handle space-separated compound constraints (all must match)
    if trimmed.contains(' ') {
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        for part in parts {
            if !semver_satisfies(version, part)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    if let Some(range) = trimmed.strip_prefix('^') {
        // Caret: compatible with (same major, >= version)
        let target = semver_parse(range)?;
        let target_major = semver_major(target);
        let ver_major = semver_major(version);
        Ok(ver_major == target_major && version >= target)
    } else if let Some(range) = trimmed.strip_prefix('~') {
        // Tilde: approximately (same major.minor, >= patch)
        let target = semver_parse(range)?;
        let target_major = semver_major(target);
        let target_minor = semver_minor(target);
        let ver_major = semver_major(version);
        let ver_minor = semver_minor(version);
        Ok(ver_major == target_major && ver_minor == target_minor && version >= target)
    } else if let Some(range) = trimmed.strip_prefix(">=") {
        let target = semver_parse(range.trim())?;
        Ok(version >= target)
    } else if let Some(range) = trimmed.strip_prefix("<=") {
        let target = semver_parse(range.trim())?;
        Ok(version <= target)
    } else if let Some(range) = trimmed.strip_prefix('>') {
        let target = semver_parse(range.trim())?;
        Ok(version > target)
    } else if let Some(range) = trimmed.strip_prefix('<') {
        let target = semver_parse(range.trim())?;
        Ok(version < target)
    } else {
        // Exact match (with optional = prefix)
        let ver_str = trimmed.strip_prefix('=').unwrap_or(trimmed).trim();
        let target = semver_parse(ver_str)?;
        Ok(version == target)
    }
}

/// Increments the major version, resets minor and patch to 0.
pub fn semver_increment_major(packed: u64) -> u64 {
    let major = ((packed >> MAJOR_SHIFT) & COMPONENT_MASK) + 1;
    (major << MAJOR_SHIFT) | PRE_FLAG // release, minor=0, patch=0
}

/// Increments the minor version, resets patch to 0.
pub fn semver_increment_minor(packed: u64) -> u64 {
    let major = (packed >> MAJOR_SHIFT) & COMPONENT_MASK;
    let minor = ((packed >> MINOR_SHIFT) & COMPONENT_MASK) + 1;
    (major << MAJOR_SHIFT) | (minor << MINOR_SHIFT) | PRE_FLAG
}

/// Increments the patch version.
pub fn semver_increment_patch(packed: u64) -> u64 {
    let major = (packed >> MAJOR_SHIFT) & COMPONENT_MASK;
    let minor = (packed >> MINOR_SHIFT) & COMPONENT_MASK;
    let patch = ((packed >> PATCH_SHIFT) & COMPONENT_MASK) + 1;
    (major << MAJOR_SHIFT) | (minor << MINOR_SHIFT) | (patch << PATCH_SHIFT) | PRE_FLAG
}

// ---------------------------------------------------------------------------
// Full precedence parsing, keeps the prerelease tag the packed form drops
// ---------------------------------------------------------------------------

/// One dot separated prerelease identifier. Numeric identifiers compare
/// numerically and sort before alphanumeric ones
#[derive(Debug, Clone, PartialEq, Eq)]
enum PreIdent {
    Numeric(u64),
    Alpha(String),
}

fn invalid_version(text: &str) -> ZyronError {
    ZyronError::InvalidParameter {
        name: "version".to_string(),
        value: text.to_string(),
    }
}

/// Parses a version string into components plus the raw prerelease tag.
/// Build metadata after '+' is accepted and ignored, it carries no
/// precedence
fn parse_full(text: &str) -> Result<(u64, u64, u64, Option<String>)> {
    let trimmed = text.trim();
    let trimmed = trimmed.strip_prefix('v').unwrap_or(trimmed);
    let without_build = match trimmed.find('+') {
        Some(idx) => &trimmed[..idx],
        None => trimmed,
    };
    let (version_part, pre) = match without_build.find('-') {
        Some(idx) => (
            &without_build[..idx],
            Some(without_build[idx + 1..].to_string()),
        ),
        None => (without_build, None),
    };
    if let Some(tag) = &pre {
        if tag.is_empty() || tag.split('.').any(|id| id.is_empty()) {
            return Err(invalid_version(text));
        }
        let valid_chars = tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.');
        if !valid_chars {
            return Err(invalid_version(text));
        }
    }
    let parts: Vec<&str> = version_part.split('.').collect();
    if parts.len() != 3 {
        return Err(invalid_version(text));
    }
    let mut nums = [0u64; 3];
    for (slot, part) in nums.iter_mut().zip(parts.iter()) {
        if part.is_empty() || !part.chars().all(|c| c.is_ascii_digit()) {
            return Err(invalid_version(text));
        }
        *slot = part.parse::<u64>().map_err(|_| invalid_version(text))?;
    }
    Ok((nums[0], nums[1], nums[2], pre))
}

fn prerelease_idents(tag: &str) -> Vec<PreIdent> {
    tag.split('.')
        .map(|id| {
            if !id.is_empty() && id.chars().all(|c| c.is_ascii_digit()) {
                match id.parse::<u64>() {
                    Ok(n) => PreIdent::Numeric(n),
                    Err(_) => PreIdent::Alpha(id.to_string()),
                }
            } else {
                PreIdent::Alpha(id.to_string())
            }
        })
        .collect()
}

fn compare_prerelease(a: &[PreIdent], b: &[PreIdent]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let ord = match (ai, bi) {
            (PreIdent::Numeric(x), PreIdent::Numeric(y)) => x.cmp(y),
            (PreIdent::Numeric(_), PreIdent::Alpha(_)) => Ordering::Less,
            (PreIdent::Alpha(_), PreIdent::Numeric(_)) => Ordering::Greater,
            (PreIdent::Alpha(x), PreIdent::Alpha(y)) => x.cmp(y),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

/// Full SemVer precedence over parsed components. A prerelease sorts before
/// its release, prerelease tags compare identifier by identifier
fn compare_full(
    a: &(u64, u64, u64, Option<String>),
    b: &(u64, u64, u64, Option<String>),
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let core = (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2));
    if core != Ordering::Equal {
        return core;
    }
    match (&a.3, &b.3) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (Some(x), Some(y)) => compare_prerelease(&prerelease_idents(x), &prerelease_idents(y)),
    }
}

/// Extracts the prerelease tag from a version string, None for a release
pub fn semver_prerelease(text: &str) -> Result<Option<String>> {
    let (_, _, _, pre) = parse_full(text)?;
    Ok(pre)
}

/// Sorts version strings by full SemVer precedence. Prerelease identifiers
/// follow the spec ordering, 1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta <
/// 1.0.0. Any invalid version fails the call naming the offending value
pub fn semver_sort(versions: &[&str]) -> Result<Vec<String>> {
    let mut parsed = Vec::with_capacity(versions.len());
    for v in versions {
        parsed.push((parse_full(v)?, v.to_string()));
    }
    parsed.sort_by(|a, b| compare_full(&a.0, &b.0));
    Ok(parsed.into_iter().map(|(_, original)| original).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let v = semver_parse("1.2.3").unwrap();
        assert_eq!(semver_major(v), 1);
        assert_eq!(semver_minor(v), 2);
        assert_eq!(semver_patch(v), 3);
        assert!(!semver_is_prerelease(v));
    }

    #[test]
    fn test_parse_prerelease() {
        let v = semver_parse("1.0.0-rc.1").unwrap();
        assert_eq!(semver_major(v), 1);
        assert_eq!(semver_minor(v), 0);
        assert_eq!(semver_patch(v), 0);
        assert!(semver_is_prerelease(v));
    }

    #[test]
    fn test_parse_v_prefix() {
        let v = semver_parse("v2.0.0").unwrap();
        assert_eq!(semver_major(v), 2);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(semver_parse("1.2").is_err());
        assert!(semver_parse("abc").is_err());
        assert!(semver_parse("").is_err());
    }

    #[test]
    fn test_format_roundtrip() {
        let v = semver_parse("1.2.3").unwrap();
        assert_eq!(semver_format(v), "1.2.3");
    }

    #[test]
    fn test_format_prerelease() {
        let v = semver_parse("1.0.0-beta").unwrap();
        assert_eq!(semver_format(v), "1.0.0-pre");
    }

    #[test]
    fn test_ordering_basic() {
        let v1 = semver_parse("1.0.0").unwrap();
        let v2 = semver_parse("2.0.0").unwrap();
        assert!(v1 < v2);
        assert_eq!(semver_compare(v1, v2), -1);
    }

    #[test]
    fn test_ordering_numeric_not_lexicographic() {
        let v1 = semver_parse("1.2.3").unwrap();
        let v2 = semver_parse("1.10.0").unwrap();
        assert!(v1 < v2, "1.2.3 should be less than 1.10.0");
    }

    #[test]
    fn test_ordering_prerelease_before_release() {
        let pre = semver_parse("1.0.0-rc.1").unwrap();
        let release = semver_parse("1.0.0").unwrap();
        assert!(pre < release, "1.0.0-rc.1 should sort before 1.0.0");
    }

    #[test]
    fn test_ordering_equal() {
        let a = semver_parse("1.2.3").unwrap();
        let b = semver_parse("1.2.3").unwrap();
        assert_eq!(semver_compare(a, b), 0);
    }

    #[test]
    fn test_satisfies_exact() {
        let v = semver_parse("1.2.3").unwrap();
        assert!(semver_satisfies(v, "1.2.3").unwrap());
        assert!(semver_satisfies(v, "=1.2.3").unwrap());
        assert!(!semver_satisfies(v, "1.2.4").unwrap());
    }

    #[test]
    fn test_satisfies_caret() {
        let v = semver_parse("1.5.0").unwrap();
        assert!(semver_satisfies(v, "^1.2.0").unwrap());
        assert!(semver_satisfies(v, "^1.0.0").unwrap());
        assert!(!semver_satisfies(v, "^2.0.0").unwrap());
        assert!(!semver_satisfies(v, "^1.6.0").unwrap());
    }

    #[test]
    fn test_satisfies_tilde() {
        let v = semver_parse("1.2.5").unwrap();
        assert!(semver_satisfies(v, "~1.2.0").unwrap());
        assert!(semver_satisfies(v, "~1.2.3").unwrap());
        assert!(!semver_satisfies(v, "~1.3.0").unwrap());
    }

    #[test]
    fn test_satisfies_gte() {
        let v = semver_parse("2.0.0").unwrap();
        assert!(semver_satisfies(v, ">=1.0.0").unwrap());
        assert!(semver_satisfies(v, ">=2.0.0").unwrap());
        assert!(!semver_satisfies(v, ">=3.0.0").unwrap());
    }

    #[test]
    fn test_satisfies_range() {
        let v = semver_parse("1.5.0").unwrap();
        assert!(semver_satisfies(v, ">=1.0.0 <2.0.0").unwrap());
        assert!(!semver_satisfies(v, ">=2.0.0 <3.0.0").unwrap());
    }

    #[test]
    fn test_increment_major() {
        let v = semver_parse("1.2.3").unwrap();
        let v2 = semver_increment_major(v);
        assert_eq!(semver_major(v2), 2);
        assert_eq!(semver_minor(v2), 0);
        assert_eq!(semver_patch(v2), 0);
    }

    #[test]
    fn test_increment_minor() {
        let v = semver_parse("1.2.3").unwrap();
        let v2 = semver_increment_minor(v);
        assert_eq!(semver_major(v2), 1);
        assert_eq!(semver_minor(v2), 3);
        assert_eq!(semver_patch(v2), 0);
    }

    #[test]
    fn test_increment_patch() {
        let v = semver_parse("1.2.3").unwrap();
        let v2 = semver_increment_patch(v);
        assert_eq!(semver_major(v2), 1);
        assert_eq!(semver_minor(v2), 2);
        assert_eq!(semver_patch(v2), 4);
    }

    #[test]
    fn test_large_version() {
        let v = semver_parse("100.200.300").unwrap();
        assert_eq!(semver_major(v), 100);
        assert_eq!(semver_minor(v), 200);
        assert_eq!(semver_patch(v), 300);
    }

    #[test]
    fn test_zero_version() {
        let v = semver_parse("0.0.0").unwrap();
        assert_eq!(semver_major(v), 0);
        assert_eq!(semver_minor(v), 0);
        assert_eq!(semver_patch(v), 0);
    }

    // semver_prerelease
    #[test]
    fn test_prerelease_present() {
        assert_eq!(
            semver_prerelease("1.0.0-rc.1").unwrap(),
            Some("rc.1".to_string())
        );
        assert_eq!(
            semver_prerelease("2.1.0-alpha").unwrap(),
            Some("alpha".to_string())
        );
    }

    #[test]
    fn test_prerelease_absent() {
        assert_eq!(semver_prerelease("1.0.0").unwrap(), None);
        assert_eq!(semver_prerelease("v1.2.3").unwrap(), None);
    }

    #[test]
    fn test_prerelease_ignores_build_metadata() {
        assert_eq!(semver_prerelease("1.0.0+build.5").unwrap(), None);
        assert_eq!(
            semver_prerelease("1.0.0-beta.2+build.5").unwrap(),
            Some("beta.2".to_string())
        );
    }

    #[test]
    fn test_prerelease_invalid_version() {
        assert!(semver_prerelease("1.2").is_err());
        assert!(semver_prerelease("1.0.0-").is_err());
        assert!(semver_prerelease("abc").is_err());
    }

    // semver_sort
    #[test]
    fn test_sort_spec_prerelease_chain() {
        let sorted = semver_sort(&[
            "1.0.0",
            "1.0.0-beta",
            "1.0.0-alpha.1",
            "1.0.0-alpha",
            "1.0.0-rc.1",
        ])
        .unwrap();
        assert_eq!(
            sorted,
            vec![
                "1.0.0-alpha".to_string(),
                "1.0.0-alpha.1".to_string(),
                "1.0.0-beta".to_string(),
                "1.0.0-rc.1".to_string(),
                "1.0.0".to_string(),
            ]
        );
    }

    #[test]
    fn test_sort_numeric_identifiers_compare_numerically() {
        let sorted = semver_sort(&["1.0.0-rc.10", "1.0.0-rc.2", "1.0.0-rc.1"]).unwrap();
        assert_eq!(
            sorted,
            vec![
                "1.0.0-rc.1".to_string(),
                "1.0.0-rc.2".to_string(),
                "1.0.0-rc.10".to_string(),
            ]
        );
    }

    #[test]
    fn test_sort_numeric_before_alpha_identifier() {
        let sorted = semver_sort(&["1.0.0-alpha", "1.0.0-1"]).unwrap();
        assert_eq!(
            sorted,
            vec!["1.0.0-1".to_string(), "1.0.0-alpha".to_string()]
        );
    }

    #[test]
    fn test_sort_core_versions_numeric() {
        let sorted = semver_sort(&["1.10.0", "1.2.3", "0.9.9", "2.0.0"]).unwrap();
        assert_eq!(
            sorted,
            vec![
                "0.9.9".to_string(),
                "1.2.3".to_string(),
                "1.10.0".to_string(),
                "2.0.0".to_string(),
            ]
        );
    }

    #[test]
    fn test_sort_invalid_names_offender() {
        let err = semver_sort(&["1.0.0", "not.a.version"]);
        match err {
            Err(ZyronError::InvalidParameter { value, .. }) => {
                assert_eq!(value, "not.a.version");
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }
}

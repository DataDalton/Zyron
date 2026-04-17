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
}

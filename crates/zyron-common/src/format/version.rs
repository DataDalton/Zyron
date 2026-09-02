//! Format version numbers and the reader window a binary carries.
//!
//! A format version is a `u16` major and a `u16` minor, encoded little
//! endian in the envelope. Major and minor are ordered lexicographically,
//! so `2.0` is newer than `1.9`. Nothing in the substrate reads a version
//! it does not have a reader for, so an unknown version is an error rather
//! than a best-effort parse

use std::fmt;

/// Major and minor version of one on-disk format
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct FormatVersion {
    pub major: u16,
    pub minor: u16,
}

impl FormatVersion {
    /// The version every format starts at when it is first registered
    pub const V1: FormatVersion = FormatVersion::new(1, 0);

    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Packs the version into the 4 envelope bytes, major first
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 4] {
        let major = self.major.to_le_bytes();
        let minor = self.minor.to_le_bytes();
        [major[0], major[1], minor[0], minor[1]]
    }

    /// Reads the version out of the 4 envelope bytes
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 4]) -> Self {
        Self {
            major: u16::from_le_bytes([bytes[0], bytes[1]]),
            minor: u16::from_le_bytes([bytes[2], bytes[3]]),
        }
    }

    /// A single sortable integer, used as a map key and for range checks
    #[inline]
    pub const fn as_u32(self) -> u32 {
        ((self.major as u32) << 16) | self.minor as u32
    }

    /// Inverse of `as_u32`
    #[inline]
    pub const fn from_u32(value: u32) -> Self {
        Self {
            major: (value >> 16) as u16,
            minor: (value & 0xFFFF) as u16,
        }
    }

    /// The next minor version, which is what a format bump within a major
    /// line produces
    pub const fn next_minor(self) -> Self {
        Self {
            major: self.major,
            minor: self.minor + 1,
        }
    }
}

impl fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl std::str::FromStr for FormatVersion {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (major, minor) = s
            .split_once('.')
            .ok_or_else(|| format!("format version `{s}` is not `<major>.<minor>`"))?;
        let major: u16 = major
            .trim()
            .parse()
            .map_err(|_| format!("format version `{s}` has a non-numeric major"))?;
        let minor: u16 = minor
            .trim()
            .parse()
            .map_err(|_| format!("format version `{s}` has a non-numeric minor"))?;
        Ok(Self { major, minor })
    }
}

/// The inclusive span of versions a binary can read for one format.
///
/// The window is bounded on purpose. Reader code for a version outside it is
/// deleted from the tree rather than kept as a shim, and the release check
/// fails the build when it lingers past its retirement date
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionWindow {
    pub oldest: FormatVersion,
    pub newest: FormatVersion,
}

impl VersionWindow {
    pub const fn new(oldest: FormatVersion, newest: FormatVersion) -> Self {
        Self { oldest, newest }
    }

    /// A window holding exactly one version, which is where every format
    /// starts
    pub const fn single(version: FormatVersion) -> Self {
        Self {
            oldest: version,
            newest: version,
        }
    }

    #[inline]
    pub fn contains(&self, version: FormatVersion) -> bool {
        version >= self.oldest && version <= self.newest
    }

    /// How many minor steps the window spans within one major line, or None
    /// when it crosses a major boundary
    pub fn minor_span(&self) -> Option<u16> {
        if self.oldest.major != self.newest.major {
            return None;
        }
        Some(self.newest.minor.saturating_sub(self.oldest.minor))
    }

    /// Every version in the window, oldest first. Only meaningful inside one
    /// major line, which is where the reader window always sits
    pub fn iter(&self) -> impl Iterator<Item = FormatVersion> + '_ {
        let oldest = self.oldest;
        let newest = self.newest;
        (oldest.as_u32()..=newest.as_u32()).filter_map(move |raw| {
            let candidate = FormatVersion::from_u32(raw);
            (candidate.major == oldest.major || candidate.major == newest.major)
                .then_some(candidate)
        })
    }
}

impl fmt::Display for VersionWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.oldest == self.newest {
            write!(f, "{}", self.oldest)
        } else {
            write!(f, "{}..={}", self.oldest, self.newest)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_round_trips_through_bytes() {
        let v = FormatVersion::new(3, 7);
        assert_eq!(FormatVersion::from_le_bytes(v.to_le_bytes()), v);
    }

    #[test]
    fn test_version_round_trips_through_u32() {
        for (major, minor) in [(0u16, 0u16), (1, 0), (1, 9), (2, 0), (65535, 65535)] {
            let v = FormatVersion::new(major, minor);
            assert_eq!(FormatVersion::from_u32(v.as_u32()), v);
        }
    }

    #[test]
    fn test_versions_order_major_first() {
        assert!(FormatVersion::new(2, 0) > FormatVersion::new(1, 9));
        assert!(FormatVersion::new(1, 10) > FormatVersion::new(1, 9));
    }

    #[test]
    fn test_window_contains_and_iterates() {
        let w = VersionWindow::new(FormatVersion::new(1, 1), FormatVersion::new(1, 3));
        assert!(w.contains(FormatVersion::new(1, 2)));
        assert!(!w.contains(FormatVersion::new(1, 0)));
        assert!(!w.contains(FormatVersion::new(2, 0)));
        let seen: Vec<_> = w.iter().collect();
        assert_eq!(
            seen,
            vec![
                FormatVersion::new(1, 1),
                FormatVersion::new(1, 2),
                FormatVersion::new(1, 3),
            ]
        );
        assert_eq!(w.minor_span(), Some(2));
    }

    #[test]
    fn test_version_parses_from_text() {
        let v: FormatVersion = "11.4".parse().expect("parses");
        assert_eq!(v, FormatVersion::new(11, 4));
        assert!("11".parse::<FormatVersion>().is_err());
        assert!("a.b".parse::<FormatVersion>().is_err());
    }
}

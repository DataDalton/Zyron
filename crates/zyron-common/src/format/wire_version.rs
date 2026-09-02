//! Wire protocol version lifecycle.
//!
//! The server speaks the current wire version and the one before it while a
//! transition is open, and nothing else. A version is negotiated once at
//! connection setup. When the transition closes, the older codec is deleted
//! from the tree rather than kept as a shim, and the registry row records
//! that it is gone

use std::fmt;

/// Where a wire protocol version sits in its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireVersionStatus {
    /// The version new connections negotiate
    Current,
    /// Still accepted while the transition is open
    Transitional,
    /// No longer accepted, the codec is gone
    Retired,
}

impl WireVersionStatus {
    pub const fn label(self) -> &'static str {
        match self {
            WireVersionStatus::Current => "current",
            WireVersionStatus::Transitional => "transitional",
            WireVersionStatus::Retired => "retired",
        }
    }

    #[inline]
    pub const fn is_accepted(self) -> bool {
        matches!(
            self,
            WireVersionStatus::Current | WireVersionStatus::Transitional
        )
    }
}

impl fmt::Display for WireVersionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One registered wire protocol version
#[derive(Debug, Clone, Copy)]
pub struct WireProtocolVersion {
    /// The number a client sends in its startup message
    pub version: u32,
    pub status: WireVersionStatus,
    /// The Zyron version that introduced it
    pub introduced_in_binary_version: &'static str,
    /// The Zyron version that stops accepting it, when one is scheduled
    pub retired_in_binary_version: Option<&'static str>,
    /// One line naming what changed at this version
    pub notes: &'static str,
}

inventory::collect!(WireProtocolVersion);

/// Why a negotiation failed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireVersionError {
    /// The client asked for a version the server has never spoken
    Unknown { requested: u32, current: u32 },
    /// The client asked for a version whose transition has closed
    Retired {
        requested: u32,
        current: u32,
        retired_in: Option<&'static str>,
    },
    /// The client asked for a version newer than this server speaks
    TooNew { requested: u32, current: u32 },
}

impl fmt::Display for WireVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireVersionError::Unknown { requested, current } => write!(
                f,
                "wire protocol version {requested} is not one this server speaks. \
                 The current version is {current}"
            ),
            WireVersionError::Retired {
                requested,
                current,
                retired_in,
            } => match retired_in {
                Some(version) => write!(
                    f,
                    "wire protocol version {requested} was retired in Zyron {version}. \
                     Update the driver to speak version {current}"
                ),
                None => write!(
                    f,
                    "wire protocol version {requested} is retired. Update the driver to \
                     speak version {current}"
                ),
            },
            WireVersionError::TooNew { requested, current } => write!(
                f,
                "wire protocol version {requested} is newer than this server speaks. \
                 The current version is {current}"
            ),
        }
    }
}

impl std::error::Error for WireVersionError {}

/// The loaded wire protocol registry
#[derive(Debug, Default)]
pub struct WireVersionRegistry {
    versions: Vec<WireProtocolVersion>,
}

impl WireVersionRegistry {
    pub fn load() -> WireVersionRegistry {
        let versions: Vec<WireProtocolVersion> = inventory::iter::<WireProtocolVersion>
            .into_iter()
            .copied()
            .collect();
        WireVersionRegistry::from_versions(versions)
    }

    pub fn from_versions(mut versions: Vec<WireProtocolVersion>) -> WireVersionRegistry {
        versions.sort_by_key(|v| v.version);
        WireVersionRegistry { versions }
    }

    pub fn versions(&self) -> &[WireProtocolVersion] {
        &self.versions
    }

    /// The version new connections negotiate
    pub fn current(&self) -> Option<&WireProtocolVersion> {
        self.versions
            .iter()
            .find(|v| v.status == WireVersionStatus::Current)
    }

    /// The version number new connections negotiate, 0 when none is
    /// registered
    pub fn current_version(&self) -> u32 {
        self.current().map(|v| v.version).unwrap_or(0)
    }

    /// Every version a client may ask for right now
    pub fn accepted(&self) -> Vec<u32> {
        self.versions
            .iter()
            .filter(|v| v.status.is_accepted())
            .map(|v| v.version)
            .collect()
    }

    /// Resolves what a connection actually speaks
    pub fn negotiate(&self, requested: u32) -> Result<u32, WireVersionError> {
        let current = self.current_version();
        match self.versions.iter().find(|v| v.version == requested) {
            Some(entry) if entry.status.is_accepted() => Ok(entry.version),
            Some(entry) => Err(WireVersionError::Retired {
                requested,
                current,
                retired_in: entry.retired_in_binary_version,
            }),
            None if requested > current => Err(WireVersionError::TooNew { requested, current }),
            None => Err(WireVersionError::Unknown { requested, current }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> WireVersionRegistry {
        WireVersionRegistry::from_versions(vec![
            WireProtocolVersion {
                version: 1,
                status: WireVersionStatus::Transitional,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: None,
                notes: "the first wire version",
            },
            WireProtocolVersion {
                version: 2,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.11.0",
                retired_in_binary_version: None,
                notes: "adds the negotiated feature list",
            },
        ])
    }

    #[test]
    fn test_both_versions_negotiate_during_the_transition() {
        let registry = registry();
        assert_eq!(registry.current_version(), 2);
        assert_eq!(registry.accepted(), vec![1, 2]);
        assert_eq!(registry.negotiate(1).expect("accepts v1"), 1);
        assert_eq!(registry.negotiate(2).expect("accepts v2"), 2);
    }

    #[test]
    fn test_retired_version_is_refused_after_cutover() {
        let registry = WireVersionRegistry::from_versions(vec![
            WireProtocolVersion {
                version: 1,
                status: WireVersionStatus::Retired,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: Some("0.12.0"),
                notes: "retired at the cutover",
            },
            WireProtocolVersion {
                version: 2,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.11.0",
                retired_in_binary_version: None,
                notes: "current",
            },
        ]);
        assert_eq!(registry.accepted(), vec![2]);
        let err = registry.negotiate(1).expect_err("refuses v1");
        assert!(err.to_string().contains("retired in Zyron 0.12.0"), "{err}");
        assert!(err.to_string().contains("version 2"), "{err}");
    }

    #[test]
    fn test_unknown_and_too_new_versions_are_named() {
        let registry = registry();
        assert!(matches!(
            registry.negotiate(99),
            Err(WireVersionError::TooNew { requested: 99, .. })
        ));
        let sparse = WireVersionRegistry::from_versions(vec![WireProtocolVersion {
            version: 5,
            status: WireVersionStatus::Current,
            introduced_in_binary_version: "0.11.0",
            retired_in_binary_version: None,
            notes: "current",
        }]);
        assert!(matches!(
            sparse.negotiate(3),
            Err(WireVersionError::Unknown { requested: 3, .. })
        ));
    }
}
